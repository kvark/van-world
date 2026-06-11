//! Toroidal MultiDiffusion: generate a canvas of arbitrary size (both
//! axes wrap) with a fixed-tile diffusion model by averaging per-tile
//! v-predictions under a raised-cosine window at every DDIM step.
//!
//! The model session must be built with batch 2: sample 0 runs the target
//! class, sample 1 the null class, combined via classifier-free guidance.

use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;

use meganeura::runtime::Session;

use crate::diffusion;
use crate::model::UNetConfig;
use crate::training::NULL_CLASS;

pub struct TiledOpts {
    pub steps: usize,
    pub guidance: f32,
    pub overlap: usize,
    pub seed: u64,
}

pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    /// CHW, both axes toroidal.
    pub data: Vec<f32>,
}

impl Canvas {
    pub fn new(channels: usize, width: usize, height: usize) -> Self {
        Canvas {
            width,
            height,
            channels,
            data: vec![0.0; channels * width * height],
        }
    }

    pub fn plane(&self, c: usize) -> &[f32] {
        &self.data[c * self.width * self.height..(c + 1) * self.width * self.height]
    }

    /// Nearest-neighbor upscale by `factor`.
    pub fn upsample(&self, factor: usize) -> Canvas {
        let mut out = Canvas::new(self.channels, self.width * factor, self.height * factor);
        for c in 0..self.channels {
            for y in 0..out.height {
                for x in 0..out.width {
                    out.data[(c * out.height + y) * out.width + x] =
                        self.data[(c * self.height + y / factor) * self.width + x / factor];
                }
            }
        }
        out
    }
}

fn tile_starts(extent: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if extent <= tile {
        assert_eq!(extent, tile, "canvas smaller than tile");
        return vec![0];
    }
    let stride = tile - overlap;
    let n = extent.div_ceil(stride);
    (0..n).map(|i| i * stride % extent).collect()
}

fn window(tile: usize) -> Vec<f32> {
    // Raised cosine in each axis, strictly positive.
    let w1: Vec<f32> = (0..tile)
        .map(|i| {
            let u = (i as f32 + 0.5) / tile as f32;
            0.05 + 0.95 * (std::f32::consts::PI * u).sin().powi(2)
        })
        .collect();
    let mut w = vec![0.0; tile * tile];
    for y in 0..tile {
        for x in 0..tile {
            w[y * tile + x] = w1[y] * w1[x];
        }
    }
    w
}

/// Generate `data_channels` of a toroidal canvas. `cond` (if present) is a
/// full-size canvas whose channels are appended to every model input tile.
pub fn generate(
    session: &mut Session,
    cfg: &UNetConfig,
    canvas_w: usize,
    canvas_h: usize,
    cond: Option<&Canvas>,
    class: u32,
    opts: &TiledOpts,
) -> Canvas {
    assert_eq!(cfg.batch, 2, "tiled generation expects a batch-2 session");
    let tile = cfg.resolution as usize;
    let dch = cfg.data_channels as usize;
    let cch = cfg.cond_channels as usize;
    if let Some(c) = cond {
        assert_eq!(c.channels, cch);
        assert_eq!((c.width, c.height), (canvas_w, canvas_h));
    } else {
        assert_eq!(cch, 0);
    }
    let plane = tile * tile;
    let pair_size = (dch + cch) * plane;

    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut canvas = Canvas::new(dch, canvas_w, canvas_h);
    for v in canvas.data.iter_mut() {
        *v = rng.sample(StandardNormal);
    }

    let xs = tile_starts(canvas_w, tile, opts.overlap);
    let ys = tile_starts(canvas_h, tile, opts.overlap);
    let win = window(tile);
    let classes = [class, NULL_CLASS];

    let mut x_input = vec![0.0f32; 2 * pair_size];
    let mut t_emb = vec![0.0f32; 2 * cfg.t_dim];
    let mut v_acc = vec![0.0f32; canvas.data.len()];
    let mut w_acc = vec![0.0f32; canvas_w * canvas_h];

    let t_start = std::time::Instant::now();
    for i in 0..opts.steps {
        let t = 1.0 - i as f32 / opts.steps as f32;
        let t_prev = 1.0 - (i + 1) as f32 / opts.steps as f32;
        for b in 0..2 {
            diffusion::timestep_embedding(t, cfg.t_dim, &mut t_emb[b * cfg.t_dim..(b + 1) * cfg.t_dim]);
        }
        v_acc.fill(0.0);
        w_acc.fill(0.0);

        for &ty in &ys {
            for &tx in &xs {
                // Gather the tile (wrapped) into both CFG batch slots.
                for c in 0..dch {
                    for dy in 0..tile {
                        let gy = (ty + dy) % canvas_h;
                        for dx in 0..tile {
                            let gx = (tx + dx) % canvas_w;
                            let v = canvas.data[(c * canvas_h + gy) * canvas_w + gx];
                            x_input[c * plane + dy * tile + dx] = v;
                        }
                    }
                }
                if let Some(cnd) = cond {
                    for c in 0..cch {
                        for dy in 0..tile {
                            let gy = (ty + dy) % canvas_h;
                            for dx in 0..tile {
                                let gx = (tx + dx) % canvas_w;
                                x_input[(dch + c) * plane + dy * tile + dx] =
                                    cnd.data[(c * canvas_h + gy) * canvas_w + gx];
                            }
                        }
                    }
                }
                x_input.copy_within(0..pair_size, pair_size);

                session.set_input("x", &x_input);
                session.set_input("t_emb", &t_emb);
                session.set_input_u32("class", &classes);
                session.step();
                session.wait();
                let v = session.read_output(2 * dch * plane);

                // Guided v, windowed scatter-add back to the canvas.
                for dy in 0..tile {
                    let gy = (ty + dy) % canvas_h;
                    for dx in 0..tile {
                        let gx = (tx + dx) % canvas_w;
                        let w = win[dy * tile + dx];
                        for c in 0..dch {
                            let vc = v[dch * plane + c * plane + dy * tile + dx];
                            let vg = vc + opts.guidance * (v[c * plane + dy * tile + dx] - vc);
                            v_acc[(c * canvas_h + gy) * canvas_w + gx] += w * vg;
                        }
                        w_acc[gy * canvas_w + gx] += w;
                    }
                }
            }
        }

        for c in 0..dch {
            for p in 0..canvas_w * canvas_h {
                v_acc[c * canvas_w * canvas_h + p] /= w_acc[p];
            }
        }
        let (a, s) = diffusion::alpha_sigma(t);
        let (ap, sp) = diffusion::alpha_sigma(t_prev);
        for (x_t, &vg) in canvas.data.iter_mut().zip(&v_acc) {
            let (x0, eps) = diffusion::from_v(*x_t, vg, a, s);
            *x_t = ap * x0 + sp * eps;
        }
        if (i + 1) % 10 == 0 {
            println!(
                "  ddim step {}/{} ({:.0}s elapsed)",
                i + 1,
                opts.steps,
                t_start.elapsed().as_secs_f64()
            );
        }
    }
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_starts_wrap() {
        assert_eq!(tile_starts(256, 256, 64), vec![0]);
        let s = tile_starts(512, 256, 64);
        assert_eq!(s, vec![0, 192, 384 % 512]);
        let s = tile_starts(2048, 128, 32);
        assert!(s.len() == 22 && s.iter().all(|&v| v < 2048));
    }

    #[test]
    fn window_positive() {
        let w = window(16);
        assert!(w.iter().all(|&v| v > 0.0));
    }
}
