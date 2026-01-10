# lunar-navigation

## Objective Overview
Individually saved work from STAC. 
Goal: Assist navigation of lunar rover through computer vision.

## Data
Artificial Lunar Landscape Dataset:
-roughly different 10,000 rendered images of lunar surfaces
-csv file with bounding box dimensions for each object; varies from 0 to multiple objects per image

## Initial Analysis
Initial analysis entailed rock classification and object detection using PyTorch + Faster R-CNN. Work saved in prev-rcnn/. Final model in prev-rcnn/lunar_object_detection.ipynb

## Final Analysis
To assist with navigation on lunar surfaces, a ResNet18 CNN + MiDaS depth estimation computer vision pipeline was developed.

In order to dramatically accelerating data processing and more specifically model training through Google Colab's GPUs, all relevant classes, functions, and model refinement occurred in Google Colab notebook: **src/lunar_nav.ipynb**. A more detailed description of analysis, steps involved, etc is included in this notebook.


# Citations

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