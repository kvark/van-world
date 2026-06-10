use std::path::Path;
use vangers::level::{
    DELTA_BITS, DELTA_MASK, DOUBLE_LEVEL, LevelConfig, LevelData, TerrainBits, load_vmc, load_vmp,
};

pub const NUM_TERRAINS: u8 = 8;
pub const MAT_BITS: usize = 3;

/// Lossless per-texel decoding of a Vangers level, minus the baked shadow
/// bits (2 and 7), which we drop deliberately: lighting is recomputed.
///
/// Unlike the PNG layer codec in vange-rs, dual cells with delta == 0 are
/// preserved via the explicit `dual` mask.
pub struct World {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub h0: Vec<u8>,
    pub h1: Vec<u8>,
    /// Raw 4-bit delta (0..=15), unscaled.
    pub delta: Vec<u8>,
    /// 1 where the texel belongs to a double-level pair.
    pub dual: Vec<u8>,
    pub m0: Vec<u8>,
    pub m1: Vec<u8>,
}

impl World {
    pub fn load(ini_path: &Path, name: &str) -> Self {
        let config = LevelConfig::load(ini_path);
        assert_eq!(config.terrains.len(), NUM_TERRAINS as usize);
        let size = (config.size.0.as_value(), config.size.1.as_value());
        let data = if config.is_compressed {
            load_vmc(&config.path_data.with_extension("vmc"), size)
        } else {
            load_vmp(&config.path_data.with_extension("vmp"), size)
        };
        Self::from_level_data(name, &data)
    }

    pub fn from_level_data(name: &str, data: &LevelData) -> Self {
        let bits = TerrainBits::new(NUM_TERRAINS);
        let (width, height) = (data.size.0 as usize, data.size.1 as usize);
        let total = width * height;
        assert_eq!(total % 2, 0);
        let mut w = World {
            name: name.to_string(),
            width,
            height,
            h0: vec![0; total],
            h1: vec![0; total],
            delta: vec![0; total],
            dual: vec![0; total],
            m0: vec![0; total],
            m1: vec![0; total],
        };
        for i in (0..total).step_by(2) {
            let (ma, mb) = (data.meta[i], data.meta[i + 1]);
            if ma & DOUBLE_LEVEL != 0 {
                let d = ((ma & DELTA_MASK) << DELTA_BITS) | (mb & DELTA_MASK);
                for j in i..i + 2 {
                    w.h0[j] = data.height[i];
                    w.h1[j] = data.height[i + 1];
                    w.delta[j] = d;
                    w.dual[j] = 1;
                    w.m0[j] = bits.read(ma);
                    w.m1[j] = bits.read(mb);
                }
            } else {
                for (j, m) in [(i, ma), (i + 1, mb)] {
                    w.h0[j] = data.height[j];
                    w.h1[j] = data.height[j];
                    w.m0[j] = bits.read(m);
                    w.m1[j] = bits.read(m);
                }
            }
        }
        w
    }

    /// Inverse of `from_level_data`. Requires `dual`, `h0`, `h1`, `delta`,
    /// `m0`, `m1` to be constant within each horizontal texel pair where
    /// dual is set (which `from_level_data` guarantees, and the generator's
    /// decode pass must enforce).
    pub fn to_level_data(&self) -> LevelData {
        let bits = TerrainBits::new(NUM_TERRAINS);
        let total = self.width * self.height;
        let mut height = vec![0u8; total];
        let mut meta = vec![0u8; total];
        for i in (0..total).step_by(2) {
            if self.dual[i] != 0 {
                debug_assert_eq!(self.dual[i + 1], 1);
                let d = self.delta[i];
                height[i] = self.h0[i];
                height[i + 1] = self.h1[i];
                meta[i] = DOUBLE_LEVEL | bits.write(self.m0[i]) | ((d >> DELTA_BITS) & DELTA_MASK);
                meta[i + 1] = DOUBLE_LEVEL | bits.write(self.m1[i]) | (d & DELTA_MASK);
            } else {
                for j in i..i + 2 {
                    height[j] = self.h0[j];
                    meta[j] = bits.write(self.m0[j]);
                }
            }
        }
        LevelData {
            size: (self.width as i32, self.height as i32),
            height: height.into_boxed_slice(),
            meta: meta.into_boxed_slice(),
        }
    }
}

/// Load every world under a `thechain` directory, sorted by name.
pub fn load_all(thechain: &Path) -> Vec<World> {
    let mut names: Vec<String> = std::fs::read_dir(thechain)
        .expect("cannot read thechain dir")
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().into_string().ok()?;
            e.path().join("world.ini").exists().then_some(name)
        })
        .collect();
    names.sort();
    names
        .iter()
        .map(|name| World::load(&thechain.join(name).join("world.ini"), name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn level_data_roundtrip() {
        let mut rng = StdRng::seed_from_u64(7);
        let (w, h) = (16, 8);
        let total = w * h;
        let mut height = vec![0u8; total];
        let mut meta = vec![0u8; total];
        let bits = TerrainBits::new(NUM_TERRAINS);
        for i in (0..total).step_by(2) {
            if rng.r#gen::<bool>() {
                let d: u8 = rng.gen_range(0..16); // includes delta == 0 duals
                let (h0, h1) = (rng.r#gen::<u8>(), rng.r#gen::<u8>());
                height[i] = h0.min(h1);
                height[i + 1] = h0.max(h1);
                meta[i] = DOUBLE_LEVEL | bits.write(rng.gen_range(0..8)) | (d >> DELTA_BITS);
                meta[i + 1] = DOUBLE_LEVEL | bits.write(rng.gen_range(0..8)) | (d & DELTA_MASK);
            } else {
                for j in i..i + 2 {
                    height[j] = rng.r#gen();
                    meta[j] = bits.write(rng.gen_range(0..8));
                }
            }
        }
        let data = LevelData {
            size: (w as i32, h as i32),
            height: height.into_boxed_slice(),
            meta: meta.into_boxed_slice(),
        };
        let world = World::from_level_data("test", &data);
        let back = world.to_level_data();
        assert_eq!(&data.height[..], &back.height[..]);
        assert_eq!(&data.meta[..], &back.meta[..]);
    }
}
