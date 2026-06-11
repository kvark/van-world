#!/usr/bin/env bash
# Fire sample_sr at each 10k-step EMA checkpoint of the running SR train.
# Emits one line per completed eval (consumed by the session monitor).
set -u
CKPT=checkpoints/sr/ema.bin
ckpt_step() { od -A n -t u8 -j 8 -N 8 "$CKPT" 2>/dev/null | tr -d ' ' || echo 0; }

for target in 10000 20000 30000 40000 50000; do
    until [ "$(ckpt_step)" -ge "$target" ]; do sleep 60; done
    out="data/sr_samples/step$target"
    if target/release/sample_sr --ckpt "$CKPT" --out "$out" --seed 7 \
        > "$out.log" 2>&1; then
        echo "eval done: step $target -> $out ($(grep -c . "$out.log") log lines)"
    else
        echo "eval FAILED at step $target (see $out.log)"
    fi
done
echo "all checkpoint evals finished"
