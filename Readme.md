# Morphological U-Net

The project investigates whether morphological variations of the U-Net improve segmentation over the unaltered baseline, in the context of the Medical Segmentation Decathlon. We implement a standard U-Net and inject morphology at the input, with residuals as extra channels, either precomputed with a fixed structuring element or computed with a trainable structuring element by a morphological block, selected by a per task survey that ranks a library of classical filters (top/bottom-hat, gradient, reconstruction and h-/volume-dome, alternating-sequential and leveling residuals) and autoselects the top channels or custom selection. We also implement a morphological U-Net that replaces the convolution itself with morphological layers, whose stages use depthwise soft erosion/dilation followed by a 1x1 projection, in configurations that morphologise the whole network, only the high-resolution stages or only the bottleneck. All soft-morphology blocks approximate dilation/erosion via logsumexp with a temperature that is annealed during training to avoid the dead gradient problem. The pipeline runs 3-fold cross-validation with segmentation and training-cost/convergence metrics, and includes an MC-Dropout branch for predictive uncertainty.

## Setup & Run

Data available in http://medicaldecathlon.com. Place in `./data/` or add path to config.

```
pip install -r requirements.txt
```

```
python3 run_preprocessing.py
```

Baseline Model:
```
for f in 0 1 2; do
    python3 train_eval.py --tag baseline --fold $f
done
```

Static Filters:
```
for f in 0 1 2; do
    python3 train_eval.py --tag staticbank --static-auto --fold $f
done
```

Trainable Filters:
```
for f in 0 1 2; do
    python3 train_eval.py --tag morphbank --morph-bank auto --fold $f
done
```

Morphological Variants:
```
for cfg in bottleneck balanced deep; do
    for f in 0 1 2; do
        python3 train_eval.py --tag morphunet_$cfg --morph-unet $cfg --morph-k 3 --fold $f
    done
done
```

```
for t in baseline staticbank morphbank bottleneck balanced deep; do
    python3 train_eval.py --fold-mean $t
done
```

```
python3 train_eval.py --compare baseline_mean_scores.json staticbank_mean_scores.json \
    morphbank_mean_scores.json bottleneck_mean_scores.json balanced_mean_scores.json \
    deep_mean_scores.json
```

## Attribution & License

Derived from the MIC-DKFZ [`basic_unet_example`](https://github.com/MIC-DKFZ/basic_unet_example). 

Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Licensed under the Apache License 2.0. 

This is a modified derivative work, original per-file copyright headers are retained. [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## References

[1] Antonelli, M., et al. "The Medical Segmentation Decathlon." Nature Communications, 2022.

[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation. " International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.

[3] Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation. "International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.
