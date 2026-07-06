#!/usr/bin/env python3
#
# Live training monitor — independent of the training terminal.
#
# train_eval.py only writes its *_train.json at the very end of a run, so mid-run
# there is no log to tail. This instead polls the rolling *_last.pth checkpoint
# (which stores epoch / best_val / best_epoch) plus nvidia-smi, and prints one
# refreshing status line per tag. Run it in ANY terminal:
#
#   python3 watch.py mpm_full_l2                 # watch one tag, fold 0
#   python3 watch.py mpm_full_l2 mpm_full_l1     # watch several at once
#   python3 watch.py mpm_full_l2 --fold 0 1 2    # watch multiple folds
#   python3 watch.py mpm_full_l2 --interval 30   # slower refresh (default 15s)
#
import argparse, glob, os, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def gpu_line():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True).strip()
        used, total, util, temp = [x.strip() for x in out.split(",")]
        return f"GPU {used}/{total}MB  {util}% util  {temp}C"
    except Exception:
        return "GPU n/a"


def ckpt_status(tag, fold):
    path = os.path.join(RESULTS, f"{tag}_f{fold}_last.pth")
    if not os.path.exists(path):
        return f"{tag} f{fold}: (no checkpoint yet)"
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
        ep, bv, be = c.get("epoch"), c.get("best_val"), c.get("best_epoch")
        age = time.time() - os.path.getmtime(path)
        stale = "  <-- stale?" if age > 600 else ""
        return (f"{tag} f{fold}: epoch {ep:>3}  |  best {bv:.4f} @ep{be:>3}  "
                f"|  ckpt {age:4.0f}s ago{stale}")
    except Exception as e:
        return f"{tag} f{fold}: (unreadable: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--fold", nargs="+", type=int, default=[0])
    ap.add_argument("--interval", type=int, default=15)
    a = ap.parse_args()
    try:
        while True:
            lines = [ckpt_status(t, f) for t in a.tags for f in a.fold]
            os.system("clear")
            print(time.strftime("%H:%M:%S"), "   ", gpu_line())
            print("-" * 72)
            print("\n".join(lines))
            print("-" * 72)
            print(f"(refresh {a.interval}s — Ctrl-C to quit)")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
