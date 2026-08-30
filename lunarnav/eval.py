"""Evaluation, baselines, bootstrap CIs, latency, and results rendering."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from lunarnav.constants import BOOTSTRAP_SAMPLES, LEAKY_DISCARDED_MAE, RANDOM_STATE, TARGET_NAMES


def mae_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def bootstrap_mae_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, float]]:
    """Bootstrap 95% CIs for per-target MAE over boxes."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    cis: list[dict[str, float]] = []

    for target_idx in range(y_true.shape[1]):
        boot_maes: list[float] = []
        for _ in range(n_samples):
            idx = rng.integers(0, n, size=n)
            boot_maes.append(float(np.mean(np.abs(y_true[idx, target_idx] - y_pred[idx, target_idx]))))
        low, high = np.percentile(boot_maes, [2.5, 97.5])
        cis.append(
            {
                "mae": float(np.mean(np.abs(y_true[:, target_idx] - y_pred[:, target_idx]))),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return cis


def mean_predictor(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    mean_vals = train_y.mean(axis=0)
    preds = np.tile(mean_vals, (len(test_y), 1))
    cis = bootstrap_mae_ci(test_y, preds)
    return preds, cis


def ridge_predictor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    preds = np.zeros_like(test_y)
    for target_idx in range(train_y.shape[1]):
        model = Ridge()
        model.fit(train_x_scaled, train_y[:, target_idx])
        preds[:, target_idx] = model.predict(test_x_scaled)

    cis = bootstrap_mae_ci(test_y, preds)
    return preds, cis


def resnet18_predict(
    model,
    rgb_crops: np.ndarray,
    device,
    batch_size: int = 32,
) -> np.ndarray:
    import cv2
    import torch
    from torchvision import transforms as T

    tf = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rgb_crops), batch_size):
            batch_crops = rgb_crops[start : start + batch_size]
            tensors = []
            for crop in batch_crops:
                tensors.append(tf(crop))
            batch = torch.stack(tensors).to(device)
            preds.append(model(batch).cpu().numpy())
    return np.vstack(preds)


def beats_baseline_outside_ci(model_ci: list[dict[str, float]], baseline_ci: list[dict[str, float]]) -> bool:
    """True when model CI upper bound is below baseline CI lower bound for every target."""
    return all(model_ci[i]["ci_high"] < baseline_ci[i]["ci_low"] for i in range(len(model_ci)))


def benchmark_resnet18_latency(model, rgb_crops: np.ndarray, device, n_samples: int = 100) -> dict[str, float]:
    import cv2
    import torch
    from torchvision import transforms as T

    tf = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    timings: dict[str, list[float]] = {"gpu": [], "cpu": []}
    crops = rgb_crops[:n_samples]

    for device_name in ("gpu", "cpu"):
        run_device = device if device_name == "gpu" else torch.device("cpu")
        model.to(run_device)
        model.eval()
        if run_device.type == "cuda":
            torch.cuda.synchronize()
        for crop in crops:
            tensor = tf(crop).unsqueeze(0).to(run_device)
            if run_device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(tensor)
            if run_device.type == "cuda":
                torch.cuda.synchronize()
            timings[device_name].append((time.perf_counter() - start) * 1000)

    model.to(device)
    return {key: float(np.median(values)) for key, values in timings.items()}


def benchmark_dpt_latency(image_paths: list[Path], device, n_samples: int = 100) -> dict[str, float]:
    import torch

    from lunarnav.depth import load_image, load_midas, predict_depth

    timings: dict[str, list[float]] = {"gpu": [], "cpu": []}
    paths = image_paths[:n_samples]

    for device_name in ("gpu", "cpu"):
        run_device = device if device_name == "gpu" else torch.device("cpu")
        midas, transform = load_midas(run_device)
        for image_path in paths:
            image = load_image(image_path)
            if run_device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = predict_depth(image, midas, transform, run_device)
            if run_device.type == "cuda":
                torch.cuda.synchronize()
            timings[device_name].append((time.perf_counter() - start) * 1000)
        del midas

    return {key: float(np.median(values)) for key, values in timings.items()}


def _format_ci(entry: dict[str, float]) -> str:
    return f"{entry['mae']:.4f} [{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"


def render_results_md(results: dict[str, Any]) -> str:
    """Render RESULTS.md strictly from results.json contents."""
    lines = [
        "# Results",
        "",
        "## Test MAE (per target)",
        "",
        "| Target | Mean predictor | Ridge (bbox-only) | ResNet18 (RGB-only) |",
        "|--------|----------------|-------------------|---------------------|",
    ]

    for idx, target in enumerate(TARGET_NAMES):
        mean_entry = results["mae"][target]["mean"]
        ridge_entry = results["mae"][target]["ridge"]
        resnet_entry = results["mae"][target]["resnet18"]
        lines.append(
            f"| {target} | {_format_ci(mean_entry)} | {_format_ci(ridge_entry)} | {_format_ci(resnet_entry)} |"
        )

    lines.extend(
        [
            "",
            "## Latency (median ms, batch=1)",
            "",
            "| Model | GPU | CPU |",
            "|-------|-----|-----|",
            f"| DPT-Hybrid (full frame) | {results['latency_ms']['dpt_hybrid']['gpu']:.2f} | "
            f"{results['latency_ms']['dpt_hybrid']['cpu']:.2f} |",
            f"| ResNet18 (224 crop) | {results['latency_ms']['resnet18']['gpu']:.2f} | "
            f"{results['latency_ms']['resnet18']['cpu']:.2f} |",
            "",
            f"Speedup (GPU): {results['speedup_gpu']:.1f}x",
            "",
            "## Hardware",
            "",
            f"- GPU: {results['hardware']['gpu']}",
            f"- CPU: {results['hardware']['cpu']}",
            f"- Test boxes: {results['dataset']['test_boxes']}",
            f"- Frames (post-filter): {results['dataset']['frame_count']}",
            "",
            "## Discarded leaky headline",
            "",
            f"The original pipeline reported `total_mae: {LEAKY_DISCARDED_MAE:.4f}` — "
            "**discarded due to target leakage** (depth crop and meta tensor contained regression targets). "
            "That model was not retrained in this patch.",
            "",
            "## Reproduce",
            "",
            "1. Open `colab/run_all.ipynb` in Google Colab Pro.",
            "2. Set runtime to A100 GPU.",
            "3. Add `GITHUB_TOKEN` and `KAGGLE_KEY` secrets.",
            "4. Run all cells.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_handoff_md(results: dict[str, Any]) -> str:
    """Owner-facing summary with the single verdict sentence."""
    lines = [
        "# Handoff",
        "",
        "## Numbers (test set, bootstrap 95% CI)",
        "",
    ]

    for target in TARGET_NAMES:
        lines.append(f"### {target}")
        for method in ("mean", "ridge", "resnet18"):
            entry = results["mae"][target][method]
            lines.append(
                f"- **{method}**: MAE {entry['mae']:.4f}, CI [{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"
            )
        lines.append("")

    lines.extend(
        [
            "## Latency",
            "",
            f"- DPT-Hybrid GPU: {results['latency_ms']['dpt_hybrid']['gpu']:.2f} ms",
            f"- ResNet18 GPU: {results['latency_ms']['resnet18']['gpu']:.2f} ms",
            f"- Speedup: {results['speedup_gpu']:.1f}x",
            "",
            "## Skipped / notes",
            "",
            f"- Leaky headline `total_mae: {LEAKY_DISCARDED_MAE:.4f}` quoted only as discarded; not retrained.",
            f"- `rel_size` dropped as a target (computable from bbox without a model).",
            "",
            "## Verdict",
            "",
        ]
    )

    verdict = results["verdict_sentence"]
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines)


def build_verdict_sentence(results: dict[str, Any]) -> str:
    beats = results["resnet18_beats_ridge_outside_ci"]
    if beats:
        return (
            "**ResNet18 beat the bbox-only ridge baseline outside the bootstrap CIs on both targets.** "
            "Use numbers from RESULTS.md for the resume bullet."
        )
    return (
        "**ResNet18 did NOT beat the bbox-only ridge baseline outside the bootstrap CIs.** "
        "This project should come off the resume until a stronger approach is demonstrated."
    )


def run_full_evaluation(
    train_df,
    test_df,
    image_dir: str | Path,
    depth_dir: str | Path,
    model,
    device,
    hardware: dict[str, str],
    frame_count: int,
    results_dir: str | Path,
) -> dict[str, Any]:
    """Run all baselines, latency, and write results.json + RESULTS.md."""
    from lunarnav.data import collect_split_arrays

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, _ = collect_split_arrays(train_df, image_dir, depth_dir)
    test_x, test_y, test_rgb = collect_split_arrays(test_df, image_dir, depth_dir)

    _, mean_ci = mean_predictor(train_y, test_y)
    _, ridge_ci = ridge_predictor(train_x, train_y, test_x, test_y)
    resnet_preds = resnet18_predict(model, test_rgb, device)
    resnet_ci = bootstrap_mae_ci(test_y, resnet_preds)

    mae_block: dict[str, dict[str, dict[str, float]]] = {}
    for idx, target in enumerate(TARGET_NAMES):
        mae_block[target] = {
            "mean": mean_ci[idx],
            "ridge": ridge_ci[idx],
            "resnet18": resnet_ci[idx],
        }

    image_paths = sorted(Path(image_dir).glob("*.png"))
    dpt_latency = benchmark_dpt_latency(image_paths, device)
    resnet_latency = benchmark_resnet18_latency(model, test_rgb, device)

    speedup_gpu = dpt_latency["gpu"] / resnet_latency["gpu"] if resnet_latency["gpu"] > 0 else float("inf")

    results: dict[str, Any] = {
        "hardware": hardware,
        "dataset": {"frame_count": frame_count, "test_boxes": int(len(test_y))},
        "mae": mae_block,
        "latency_ms": {"dpt_hybrid": dpt_latency, "resnet18": resnet_latency},
        "speedup_gpu": speedup_gpu,
        "discarded_leaky": {
            "total_mae": LEAKY_DISCARDED_MAE,
            "reason": "discarded due to target leakage",
        },
        "resnet18_beats_ridge_outside_ci": beats_baseline_outside_ci(resnet_ci, ridge_ci),
    }
    results["verdict_sentence"] = build_verdict_sentence(results)

    results_json = results_dir / "results.json"
    with open(results_json, "w") as handle:
        json.dump(results, handle, indent=2)

    results_md = render_results_md(results)
    (results_dir.parent / "RESULTS.md").write_text(results_md)

    handoff_md = render_handoff_md(results)
    (results_dir.parent / "HANDOFF.md").write_text(handoff_md)

    return results
