# setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Tasks
export TASK=Task01_BrainTumour
export TASK=Task02_Heart
export TASK=Task10_Colon

# preprocessing
python3 run_preprocessing.py

# experiment 1 (noise on test)
for kind in gamma contrast noise; do \
    if [ "$kind" = noise ]; then STR="0.05 0.1 0.2 0.3"; else STR="1.0 0.7 0.5 1.5 2.0"; fi; \
    for s in $STR; do for f in 0 1 2; do \
        python3 train_eval.py --tag baseline --fold $f --test-only --test-perturb $kind --perturb-strength $s; \
        python3 train_eval.py --tag convctrl --fold $f --test-only --test-perturb $kind --perturb-strength $s --morph-bank auto --conv-control; \
        python3 train_eval.py --tag morphbank --fold $f --test-only --test-perturb $kind --perturb-strength $s --morph-bank auto; \
    done; done; \
done

# experiment 2 (limited train set)
for N in 5 15 30 60; do for f in 0 1 2; do \
  python3 train_eval.py --tag baseline --fold $f --train-cases $N; \
  python3 train_eval.py --tag convctrl --fold $f --train-cases $N --morph-bank auto --conv-control; \
  python3 train_eval.py --tag morphbank --fold $f --train-cases $N --morph-bank auto; \
done; done

# staticbank
for f in 0 1 2; do
    python3 train_eval.py --tag baseline --fold $f
    python3 train_eval.py --tag staticbank --static-auto --fold $f
done

# morphbank
for f in 0 1 2; do
    python3 train_eval.py --tag morphbank --morph-bank auto --fold $f
    python3 train_eval.py --tag convctrl --morph-bank auto --conv-control --fold $f
done

# mean
for t in baseline staticbank morphbank convctrl softmorphbank; do
    python3 train_eval.py --fold-mean $t
done

# compare
python3 train_eval.py --compare baseline_mean_scores.json staticbank_mean_scores.json \
    morphbank_mean_scores.json convctrl_mean_scores.json softmorphbank_mean_scores.json
