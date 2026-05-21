# Morphological U-Net

The project investigates whether morphological variations of the U-Net improves segmentation over the unaltered baseline. This is done in the context of the Medical Segmentation Decathlon.

## Attribution & License

Derived from the MIC-DKFZ [`basic_unet_example`](https://github.com/MIC-DKFZ/basic_unet_example). Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Licensed under the Apache License 2.0. This is a modified derivative work, original per-file copyright headers are retained. [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Setup & Run

Data available in (http://medicaldecathlon.com).

Install requirements :
```
pip install -r requirements.txt
```

Preprocess data :
```
python3 run_preprocessing.py
```

Train, test and evaluate :
```
python3 train_eval.py --tag baseline --fold 0
```

Per run it writes (output stem `<tag>_f<fold>`) `<tag>_f<fold>_best.pth` / `_last.pth` (checkpoints) and `<tag>_f<fold>_scores.json` (per-label metrics).

A 5-fold cross-validated sweep is the same command looped over `--fold 0...4`.
   
## References

[1] Antonelli, M., et al. "The Medical Segmentation Decathlon." Nature Communications, 2022.

[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation. " International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.

[3] Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation. "International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.
