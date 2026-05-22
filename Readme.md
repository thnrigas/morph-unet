# Morphological U-Net

The project investigates whether morphological variations of the U-Net improves segmentation over the unaltered baseline. This is done in the context of the Medical Segmentation Decathlon. Morphological variants include top and/or bottom hat (residuals) as additional input channels, either precomputed with a fixed structuring element or computed by a morphological block with a trainable structuring element.

## Attribution & License

Derived from the MIC-DKFZ [`basic_unet_example`](https://github.com/MIC-DKFZ/basic_unet_example). 

Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Licensed under the Apache License 2.0. 

This is a modified derivative work, original per-file copyright headers are retained. [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Setup & Run

Data available in http://medicaldecathlon.com.

Install requirements :
```
pip install -r requirements.txt
```

Preprocess data :
```
python3 run_preprocessing.py
```

Train, test and evaluate :

Baseline Model :
```
python3 train_eval.py --tag baseline --fold 0
```

Static Residuals : (one or both)
```
python3 train_eval.py --tag static --tophat --bottomhat --fold 0
```

Trainable Residuals : (one or both)
```
python3 train_eval.py --tag trainable --morph-block --tophat --bottomhat --fold 0
```

Compare Results :
```
python3 train_eval.py --compare baseline_f0_scores.json static_f0_scores.json trainable_f0_scores.json
```

5-fold cross-validated sweep :
```
for f in 0 1 2 3 4; do
  python3 train_eval.py --tag baseline --fold $f
  python3 train_eval.py --tag static --tophat --bottomhat --fold $f
  python3 train_eval.py --tag trainable --morph-block --tophat --bottomhat --fold $f
done
```

Mean score over folds :
```
python3 train_eval.py --fold-mean baseline
python3 train_eval.py --fold-mean static
python3 train_eval.py --fold-mean trainable
```

Compare Results :
```
python3 train_eval.py --compare baseline_mean_scores.json static_mean_scores.json trainable_mean_scores.json
```

## References

[1] Antonelli, M., et al. "The Medical Segmentation Decathlon." Nature Communications, 2022.

[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation. " International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.

[3] Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation. "International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.
