#!/usr/bin/env bash
# Wait for the next checkpoint (step >= 10000), then restart train_sr
# resumed from it at 50% GPU duty.
set -u
CKPT=checkpoints/sr/latest.bin
# Resolve the actual trainer binary pid — `$!` after `setsid cmd &` returns
# the wrapper, not the trainer, which is how a duplicate run once survived.
PID=$(pgrep -f 'target/release/train_sr' | head -1)
[ -n "$PID" ] || { echo "no running trainer found"; exit 1; }
ckpt_step() { od -A n -t u8 -j 8 -N 8 "$CKPT" 2>/dev/null | tr -d ' ' || echo 0; }

until [ "$(ckpt_step)" -ge 10000 ]; do sleep 30; done
echo "checkpoint $(ckpt_step) reached; stopping pid $PID"
kill "$PID"
for _ in $(seq 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -0 "$PID" 2>/dev/null && { echo "old trainer did not exit"; exit 1; }

setsid nohup target/release/train_sr \
    --steps 50000 --batch 4 --res 128 --base 64 --save-every 2000 \
    --out checkpoints/sr --resume checkpoints/sr/latest.bin --duty 0.5 \
    >> logs/train_sr.log 2>&1 < /dev/null &
echo "restarted throttled trainer (pid $!) resumed from step $(ckpt_step)"
