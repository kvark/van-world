use crate::world::{MAT_BITS, NUM_TERRAINS, World};
use rand::Rng;

/// Full-resolution patch channels, all mapped to [-1, 1]:
/// 0: h0, 1: h1-h0, 2: delta, 3: dual mask, 4..7: m0 analog bits, 7..10: m1 analog bits.
pub const FULL_CHANNELS: usize = 4 + 2 * MAT_BITS;
/// Coarse (/POOL) channels: 0: mean h0, 1..4: majority m0 analog bits.
pub const COARSE_CHANNELS: usize = 1 + MAT_BITS;
pub const POOL: usize = 8;

#[derive(Copy, Clone, Debug)]
pub struct Crop {
    pub world: usize,
    pub x: usize,
    pub y: usize,
    pub flip_x: bool,
    pub flip_y: bool,
}

fn bit(v: u8, b: usize) -> f32 {
    if v >> b & 1 != 0 { 1.0 } else { -1.0 }
}

pub struct Sampler<'a> {
    worlds: &'a [World],
    cumulative_area: Vec<u64>,
}

impl<'a> Sampler<'a> {
    pub fn new(worlds: &'a [World]) -> Self {
        let mut acc = 0u64;
        let cumulative_area = worlds
            .iter()
            .map(|w| {
                acc += (w.width * w.height) as u64;
                acc
            })
            .collect();
        Sampler {
            worlds,
            cumulative_area,
        }
    }

    /// Random toroidal crop, world chosen proportionally to its area.
    pub fn sample_crop(&self, rng: &mut impl Rng) -> Crop {
        let pick = rng.gen_range(0..*self.cumulative_area.last().unwrap());
        let world = self.cumulative_area.partition_point(|&a| a <= pick);
        let w = &self.worlds[world];
        Crop {
            world,
            x: rng.gen_range(0..w.width),
            y: rng.gen_range(0..w.height),
            flip_x: rng.r#gen(),
            flip_y: rng.r#gen(),
        }
    }

    #[inline]
    fn texel(&self, crop: &Crop, dx: usize, dy: usize, size: usize) -> usize {
        let w = &self.worlds[crop.world];
        let dx = if crop.flip_x { size - 1 - dx } else { dx };
        let dy = if crop.flip_y { size - 1 - dy } else { dy };
        ((crop.y + dy) & (w.height - 1)) * w.width + ((crop.x + dx) & (w.width - 1))
    }

    /// Pack a `size`×`size` full-resolution patch as CHW into `out`.
    pub fn pack_full(&self, crop: &Crop, size: usize, out: &mut [f32]) {
        let w = &self.worlds[crop.world];
        let plane = size * size;
        assert_eq!(out.len(), FULL_CHANNELS * plane);
        for dy in 0..size {
            for dx in 0..size {
                let src = self.texel(crop, dx, dy, size);
                let dst = dy * size + dx;
                out[dst] = w.h0[src] as f32 / 127.5 - 1.0;
                // h1 < h0 happens in original data (see the disabled assert in
                // vange-rs layers.rs), so this must be signed.
                out[plane + dst] = (w.h1[src] as f32 - w.h0[src] as f32) / 127.5 - 1.0;
                out[2 * plane + dst] = w.delta[src] as f32 / 7.5 - 1.0;
                out[3 * plane + dst] = if w.dual[src] != 0 { 1.0 } else { -1.0 };
                for b in 0..MAT_BITS {
                    out[(4 + b) * plane + dst] = bit(w.m0[src], b);
                    out[(4 + MAT_BITS + b) * plane + dst] = bit(w.m1[src], b);
                }
            }
        }
    }

    /// Pack a coarse patch: `size`×`size` output texels, each pooled from a
    /// POOL×POOL full-resolution block (so the crop covers size*POOL texels).
    pub fn pack_coarse(&self, crop: &Crop, size: usize, out: &mut [f32]) {
        let w = &self.worlds[crop.world];
        let plane = size * size;
        assert_eq!(out.len(), COARSE_CHANNELS * plane);
        for dy in 0..size {
            for dx in 0..size {
                let mut h_sum = 0u32;
                let mut counts = [0u16; NUM_TERRAINS as usize];
                for py in 0..POOL {
                    for px in 0..POOL {
                        let src =
                            self.texel(crop, dx * POOL + px, dy * POOL + py, size * POOL);
                        h_sum += w.h0[src] as u32;
                        counts[w.m0[src] as usize] += 1;
                    }
                }
                let majority = counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &c)| c)
                    .unwrap()
                    .0 as u8;
                let dst = dy * size + dx;
                out[dst] = h_sum as f32 / (POOL * POOL) as f32 / 127.5 - 1.0;
                for b in 0..MAT_BITS {
                    out[(1 + b) * plane + dst] = bit(majority, b);
                }
            }
        }
    }

    /// Fill a training batch: returns world-index labels per sample.
    pub fn fill_batch(
        &self,
        rng: &mut impl Rng,
        batch: usize,
        size: usize,
        coarse: bool,
        out: &mut Vec<f32>,
        labels: &mut Vec<u32>,
    ) {
        let channels = if coarse { COARSE_CHANNELS } else { FULL_CHANNELS };
        let sample = channels * size * size;
        out.resize(batch * sample, 0.0);
        labels.clear();
        for i in 0..batch {
            let crop = self.sample_crop(rng);
            let slice = &mut out[i * sample..(i + 1) * sample];
            if coarse {
                self.pack_coarse(&crop, size, slice);
            } else {
                self.pack_full(&crop, size, slice);
            }
            labels.push(crop.world as u32);
        }
    }
}

/// Decode an analog-bit material value back to a terrain type.
pub fn decode_material(bits: [f32; MAT_BITS]) -> u8 {
    bits.iter()
        .enumerate()
        .map(|(i, &v)| if v > 0.0 { 1 << i } else { 0 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn flat_world(width: usize, height: usize) -> World {
        let total = width * height;
        World {
            name: "flat".into(),
            width,
            height,
            h0: (0..total).map(|i| (i % 256) as u8).collect(),
            h1: (0..total).map(|i| (i % 256) as u8).collect(),
            delta: vec![0; total],
            dual: vec![0; total],
            m0: (0..total).map(|i| (i % 8) as u8).collect(),
            m1: (0..total).map(|i| (i % 8) as u8).collect(),
        }
    }

    #[test]
    fn material_bits_roundtrip() {
        for t in 0..8u8 {
            let bits = [bit(t, 0), bit(t, 1), bit(t, 2)];
            assert_eq!(decode_material(bits), t);
        }
    }

    #[test]
    fn toroidal_wrap_and_shapes() {
        let world = flat_world(64, 32);
        let worlds = [world];
        let sampler = Sampler::new(&worlds);
        let crop = Crop {
            world: 0,
            x: 60,
            y: 30,
            flip_x: false,
            flip_y: false,
        };
        let size = 16;
        let mut out = vec![0.0; FULL_CHANNELS * size * size];
        sampler.pack_full(&crop, size, &mut out);
        // texel (0,0) of the crop is world texel (60, 30)
        let expected = ((30 * 64 + 60) % 256) as f32 / 127.5 - 1.0;
        assert_eq!(out[0], expected);
        // texel (4, 2) wraps to world texel (0, 0)
        let dst = 2 * size + 4;
        assert_eq!(out[dst], 0.0 / 127.5 - 1.0);

        let mut coarse = vec![0.0; COARSE_CHANNELS * 4 * 4];
        sampler.pack_coarse(&crop, 4, &mut coarse);
    }

    #[test]
    fn batch_fill() {
        let worlds = [flat_world(64, 32), flat_world(32, 32)];
        let sampler = Sampler::new(&worlds);
        let mut rng = StdRng::seed_from_u64(3);
        let (mut out, mut labels) = (Vec::new(), Vec::new());
        sampler.fill_batch(&mut rng, 4, 8, false, &mut out, &mut labels);
        assert_eq!(out.len(), 4 * FULL_CHANNELS * 64);
        assert_eq!(labels.len(), 4);
        assert!(labels.iter().all(|&l| l < 2));
    }
}
