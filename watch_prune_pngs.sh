#!/usr/bin/env bash
#
# Regenerate the per-model prune-table PNGs (slides/figs/prune_*.png) every time a new pruning run
# writes its _prune.json, so the figures stay current as the fbnew k03/k05 sweep and the random-global
# convsep sweep progress. CPU-only (matplotlib) -> does not touch the GPU jobs. Runs until both prune
# drivers exit, then does a final regen. Signature = (#_prune.json : newest mtime).
#
set -uo pipefail
cd "$(dirname "$0")"
sig(){ local n m; n=$(ls results/*_prune.json 2>/dev/null | wc -l)
       m=$(ls -t results/*_prune.json 2>/dev/null | head -1 | xargs -r stat -c %Y); echo "$n:${m:-0}"; }
regen(){ python3 slides/figs/prune_tables.py >/dev/null 2>&1 && echo "$(date '+%T') regenerated prune PNGs"; }

prev="$(sig)"; regen
while pgrep -f "fbnew_extremes.sh" >/dev/null 2>&1 || pgrep -f "random_convsep.sh" >/dev/null 2>&1; do
  sleep 45
  cur="$(sig)"
  if [[ "$cur" != "$prev" ]]; then regen; prev="$cur"; fi
done
# drivers finished -> catch the last results
sleep 5; regen
echo "prune-PNG watcher done $(date '+%F %T')"
