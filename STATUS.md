# Status — 2026-06-10

Generative pipeline for Vangers worlds: meganeura-trained cascaded pixel
diffusion (stage 1: /8 coarse structure, 4ch; stage 2: ×8 SR, 10ch),
toroidal MultiDiffusion generation, decode to playable VMP levels.

## Where things stand

- All code phases are complete and unit-tested: dataset codec (`world.rs`,
  `sampler.rs`), trainers (`train_sr`, `train_struct` over `training.rs`),
  inference (`sample_sr`, `tiled.rs`, `decode.rs`, `world_gen`).
- First SR training run (50k steps, batch 4, res 128, base 64) was aborted.
  **Evidence of a training bug**: loss plateaued at 0.7–0.8 (v-variance
  baseline ≈ 0.95, so barely better than predicting the conditional mean)
  through 10k steps, and DDIM samples at the 10k checkpoint were pure noise
  (`data/sr_samples/step10000/`, gitignored).
- `checkpoints/sr/` (gitignored) is **corrupted** past step 10000: two
  trainer processes briefly ran concurrently, interleaving checkpoint
  writes. Do not resume from it.

## Next steps (start here)

1. Run the overfit diagnostic — one frozen batch must drive loss → ~0:
   `cargo run --release --bin train_sr -- --overfit --steps 3000 --batch 2
   --res 64 --base 32 --lr 1e-3 --save-every 3000 --out checkpoints/overfit`
   - If loss collapses: pipeline fine, original run was undertrained/mistuned
     (try higher LR, no grad clip, longer warm-up).
   - If loss plateaus: autodiff bug in the graph. Bisect by stripping the
     model: drop FiLM conditioning (`film_shift`), then class embedding,
     then test meganeura's reference `sd_unet` training on real (non-random)
     data. Prime suspects: grad flow through `group_norm` / `upsample_2x` /
     `concat` chains, and `set_grad_clip_norm` (note the leaked
     `grad_clip_acc` buffer warning at exit).
2. Then retrain SR (`--duty 0.5` available for shared machines), eval at 10k
   checkpoints via `tools/eval_checkpoints.sh`, then `train_struct`, then
   `world_gen` end-to-end and render in vange-rs.

## Notes

- Data prerequisites on a new machine: `/x/Work/VangersData` (or pass
  `--data <path>/thechain`), sibling checkouts of `../vange-rs` and
  `../meganeura`.
- Model I/O and channel encodings are documented in `src/model.rs` and
  `src/sampler.rs` doc comments.
- Adam moments are not checkpointed; resume re-estimates them.
