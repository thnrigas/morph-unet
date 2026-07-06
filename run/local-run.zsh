# setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Task01_BrainTumour
export TASK=Task01_BrainTumour

# Task02_Heart
export TASK=Task02_Heart

# Task10_Colon
export TASK=Task10_Colon

python3 run_preprocessing.py

for f in 0 1 2; do
    python3 train_eval.py --tag baseline --fold $f
    python3 train_eval.py --tag staticbank --static-auto --fold $f
done

for f in 0 1 2; do
    python3 train_eval.py --tag morphbank --morph-bank auto --fold $f
done


# Task04_Hippocampus

# preprocessing
python3 run_preprocessing.py

# baseline
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag baseline --fold $f
done

# tophat
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag tophat --tophat --fold $f
done

# morphblock
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag morphblock --morph-block --tophat --fold $f
done

# morphloss
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag morphloss --morph-loss --fold $f
done

# mean
for t in baseline tophat morphblock morphloss; do
    python3 train_eval.py --fold-mean $t
done

# compare
python3 train_eval.py --compare baseline_mean_scores.json tophat_mean_scores.json \
    morphblock_mean_scores.json morphloss_mean_scores.json
