#!/usr/bin/env bash
# Run the whole experimental matrix on this machine, one job at a time.
#   caffeinate -dims bash scripts/run_local.sh
# Each step logs to logs/<step>.log; finished steps are listed in logs/done.txt and
# skipped on re-run, so the queue can be restarted after an interruption.
# Timing experiments (Exp 4, 5) run last, with nothing else on the machine.
# Exp 6 (online service) needs the deployed endpoint and is not part of this queue.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
touch logs/done.txt
CATEGORIES="sheet_metal can"
ALL="patchcore cae resnet50 efficientnet_b0 vit"

# Memory on an 8 GB machine: micro-batches with gradient accumulation keep the
# effective batch at 16 (README, decision D15).
mem_args() {
  case "$1" in
    vit) echo "--set train.batch_size=4 --set train.accumulation_steps=4" ;;
    resnet50|efficientnet_b0) echo "--set train.batch_size=8 --set train.accumulation_steps=2" ;;
    *) echo "" ;;
  esac
}

step() {
  local name=$1; shift
  if grep -qx "$name" logs/done.txt; then echo "skip  $name"; return 0; fi
  echo "[$(date '+%F %T')] start $name"
  if "$@" > "logs/$name.log" 2>&1; then
    echo "$name" >> logs/done.txt
    echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAILED $name (see logs/$name.log)"
  fi
}

echo "== training"
for cat in $CATEGORIES; do
  for m in patchcore cae resnet50 efficientnet_b0 vit; do
    for s in 0 1 2; do
      if [ -f "artifacts/$cat/$m/seed$s/calibration.json" ]; then echo "skip  train_${m}_${cat}_seed$s (exists)"; continue; fi
      # shellcheck disable=SC2046
      step "train_${m}_${cat}_seed$s" $PY scripts/02_train.py --model "$m" --category "$cat" --seed "$s" $(mem_args "$m")
    done
  done
done

echo "== evaluation (Exp 1, Exp 2+3)"
for cat in $CATEGORIES; do
  for m in $ALL; do
    # shellcheck disable=SC2046
    step "exp1_${m}_${cat}" $PY scripts/03_eval_nominal.py --model "$m" --category "$cat" $(mem_args "$m")
    # shellcheck disable=SC2046
    step "exp2_${m}_${cat}" $PY scripts/04_eval_reliability.py --model "$m" --category "$cat" $(mem_args "$m")
  done
done

echo "== timing (Exp 4, Exp 5)"
for cat in $CATEGORIES; do
  for m in $ALL; do
    step "exp4_${m}_${cat}" $PY scripts/05_bench_latency.py --model "$m" --category "$cat"
    step "exp5_${m}_${cat}" $PY scripts/06_stream.py --model "$m" --category "$cat"
  done
done

echo "== figures and tables"
step "figures" $PY scripts/08_make_figures.py
echo "queue finished; failures (if any) are marked FAILED above"
