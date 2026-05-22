# setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# preprocessing
python3 run_preprocessing.py

# baseline
for f in 0 1 2; do python3 train_eval.py --tag baseline --fold $f; done

# static top-hat
for f in 0 1 2; do python3 train_eval.py --tag statictop --fold $f --tophat; done

# static top-hat + bottom-hat
for f in 0 1 2; do python3 train_eval.py --tag staticboth --fold $f --tophat --bottomhat; done

# baseline + morph loss
for f in 0 1 2; do python3 train_eval.py --tag baseloss --fold $f --morph-loss; done

# static top-hat + morph loss
for f in 0 1 2; do python3 train_eval.py --tag toploss --fold $f --morph-loss --tophat; done

# trainable top-hat
for f in 0 1 2; do python3 train_eval.py --tag traintop --fold $f --morph-block --tophat; done

# mean
for t in baseline statictop staticboth baseloss toploss traintop; do
    python3 train_eval.py --fold-mean $t
done

# compare
python3 train_eval.py --compare baseline_mean_scores.json statictop_mean_scores.json \
    staticboth_mean_scores.json baseloss_mean_scores.json toploss_mean_scores.json traintop_mean_scores.json
