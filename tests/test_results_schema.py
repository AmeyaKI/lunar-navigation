"""Tests for results JSON rendering (no torch weights)."""

from lunarnav.constants import LEAKY_DISCARDED_MAE
from lunarnav.eval import render_handoff_md, render_results_md


FIXTURE_RESULTS = {
    "hardware": {"gpu": "NVIDIA A100", "cpu": "Intel Xeon"},
    "dataset": {"frame_count": 8993, "test_boxes": 3364},
    "mae": {
        "rel_distance": {
            "mean": {"mae": 0.05, "ci_low": 0.04, "ci_high": 0.06},
            "ridge": {"mae": 0.03, "ci_low": 0.02, "ci_high": 0.04},
            "resnet18": {"mae": 0.035, "ci_low": 0.025, "ci_high": 0.045},
        },
        "rel_height": {
            "mean": {"mae": 0.06, "ci_low": 0.05, "ci_high": 0.07},
            "ridge": {"mae": 0.04, "ci_low": 0.03, "ci_high": 0.05},
            "resnet18": {"mae": 0.042, "ci_low": 0.032, "ci_high": 0.052},
        },
    },
    "latency_ms": {
        "dpt_hybrid": {"gpu": 120.0, "cpu": 800.0},
        "resnet18": {"gpu": 2.5, "cpu": 15.0},
    },
    "speedup_gpu": 48.0,
    "discarded_leaky": {"total_mae": LEAKY_DISCARDED_MAE, "reason": "discarded due to target leakage"},
    "resnet18_beats_ridge_outside_ci": False,
    "verdict_sentence": "ResNet18 did NOT beat ridge.",
}


def test_render_results_md_contains_numbers_and_disclaimer():
    md = render_results_md(FIXTURE_RESULTS)
    assert "0.0350" in md
    assert "0.0420" in md
    assert "discarded due to target leakage" in md
    assert f"{LEAKY_DISCARDED_MAE:.4f}" in md
    assert "colab/run_all.ipynb" in md
    assert "DPT-Hybrid" in md
    assert "ResNet18" in md


def test_render_handoff_md_contains_verdict():
    md = render_handoff_md(FIXTURE_RESULTS)
    assert "ResNet18 did NOT beat ridge." in md
    assert "rel_distance" in md
    assert "Speedup" in md
