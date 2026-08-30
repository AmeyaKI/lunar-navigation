# lunar-navigation

## Objective

Distill MiDaS DPT-Hybrid geometry estimates into a small **RGB-only** ResNet18 that could run on rover hardware. Given dataset-provided bounding boxes (not a trained detector), the model predicts relative distance and relative height pseudo-labels derived from monocular depth.

This repo prioritizes **honest, checkable claims** over impressive-sounding numbers.

## What changed and why

The original pipeline in [`src/lunar_nav_v1.ipynb`](src/lunar_nav_v1.ipynb) reported `total_mae: 0.0135`, but that metric is **not defensible**:

1. Regression targets (`rel_distance`, `rel_height`, `rel_size`) were computed from the same MiDaS depth map fed to the model.
2. `rel_distance` is the center pixel of the depth crop — literally present in the input.
3. `rel_size` was passed in the `meta` tensor and also used as a target (the model was fed its own label).
4. MiDaS produces **relative**, not metric, depth — there is no ground-truth geometry in this dataset.
5. The training loop never reset `total_loss` across epochs, so printed loss rose every epoch.

The patched pipeline removes leakage: **RGB-only** ResNet18, **two targets** (distance, height; `rel_size` dropped because box area is computable without a model), frame-level **72/8/20** splits, and evaluation against **mean** and **bbox-only ridge** baselines with bootstrap 95% CIs.

## Data

[Artificial Lunar Landscape Dataset](https://www.kaggle.com/datasets/romainpessia/artificial-lunar-rocky-landscape-dataset) — ~10,000 rendered lunar surface images with bounding-box CSV. After filtering 773 faulty frames (per dataset authors), **8,993** frames remain. Boxes are **dataset-provided**; the model does not detect objects.

## Previous work (`prev-rcnn/`)

An early attempt at rock classification and Faster R-CNN object detection lives in [`prev-rcnn/`](prev-rcnn/). It was abandoned: RCNN was too heavy for local training, and the final geometry pipeline uses **dataset-provided boxes**, not detector outputs. Do not describe this project as detecting or classifying rocks in the final pipeline.

## Running the honest evaluation

Results are **not** hard-coded. To reproduce:

1. Open [`colab/run_all.ipynb`](colab/run_all.ipynb) in Google Colab Pro.
2. Set runtime to **A100 GPU** (Runtime → Change runtime type).
3. Add Colab secrets: `GITHUB_TOKEN`, `KAGGLE_KEY` (and optionally `KAGGLE_USERNAME` if `KAGGLE_KEY` is a raw API key).
4. **Run all** cells. Depth maps, training, baselines, and latency run on Colab; results are written to `RESULTS.md` and `HANDOFF.md` and pushed back to branch `fix/leakage-distill`.

After a successful run, see [`RESULTS.md`](RESULTS.md) for MAE tables with CIs, latency, and hardware. See [`HANDOFF.md`](HANDOFF.md) for the owner verdict on whether ResNet18 beat the ridge baseline outside CIs.

## Citations

**Artificial Lunar Landscape Dataset**:
@misc{romain_pessia_prof__genya_ishigami_quentin_jodelet_2025,
	title={Artificial Lunar Landscape Dataset},
	url={https://www.kaggle.com/dsv/13263000},
	DOI={10.34740/KAGGLE/DSV/13263000},
	publisher={Kaggle},
	author={Romain Pessia and Prof. Genya Ishigami and Quentin Jodelet},
	year={2025}
}

**MiDaS Depth Estimation Model:**
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
