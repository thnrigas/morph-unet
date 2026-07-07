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
