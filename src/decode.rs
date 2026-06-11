//! Deterministic decode pass: model canvas → game-valid `World` →
//! packaged level directory openable by vange-rs.

use std::path::Path;

use crate::sampler::FULL_CHANNELS;
use crate::world::MAT_BITS;
use crate::tiled::Canvas;
use crate::world::World;

fn to_u8(v: f32) -> u8 {
    ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8
}

fn mat_from_bits(canvas: &Canvas, first_channel: usize, i0: usize, i1: usize) -> u8 {
    let plane = canvas.width * canvas.height;
    let mut m = 0u8;
    for b in 0..MAT_BITS {
        let p = &canvas.data[(first_channel + b) * plane..];
        if p[i0] + p[i1] > 0.0 {
            m |= 1 << b;
        }
    }
    m
}

/// Convert a generated full-res canvas (FULL_CHANNELS) into a `World`,
/// enforcing the encoding constraints:
/// - dual flag, heights, delta, and materials constant per horizontal pair
/// - h1 >= h0 + 1 on dual cells, delta quantized to 4 bits
pub fn canvas_to_world(canvas: &Canvas, name: &str) -> World {
    assert_eq!(canvas.channels, FULL_CHANNELS);
    let (w, h) = (canvas.width, canvas.height);
    let plane = w * h;
    let total = plane;
    let d = &canvas.data;
    let ch = |c: usize, i: usize| d[c * plane + i];

    let mut world = World {
        name: name.to_string(),
        width: w,
        height: h,
        h0: vec![0; total],
        h1: vec![0; total],
        delta: vec![0; total],
        dual: vec![0; total],
        m0: vec![0; total],
        m1: vec![0; total],
    };

    for y in 0..h {
        for x in (0..w).step_by(2) {
            let i0 = y * w + x;
            let i1 = i0 + 1;
            let dual = ch(3, i0) + ch(3, i1) > 0.0;
            if dual {
                let h0 = to_u8((ch(0, i0) + ch(0, i1)) * 0.5);
                let rel = ((ch(1, i0) + ch(1, i1)) * 0.5 + 1.0) * 127.5;
                let h1 = (h0 as f32 + rel.max(1.0)).round().clamp(0.0, 255.0) as u8;
                let h1 = h1.max(h0.saturating_add(1));
                let delta =
                    (((ch(2, i0) + ch(2, i1)) * 0.5 + 1.0) * 7.5).round().clamp(0.0, 15.0) as u8;
                let m0 = mat_from_bits(canvas, 4, i0, i1);
                let m1 = mat_from_bits(canvas, 4 + MAT_BITS, i0, i1);
                for i in [i0, i1] {
                    world.h0[i] = h0;
                    world.h1[i] = h1;
                    world.delta[i] = delta;
                    world.dual[i] = 1;
                    world.m0[i] = m0;
                    world.m1[i] = m1;
                }
            } else {
                for i in [i0, i1] {
                    let h0 = to_u8(ch(0, i));
                    let m = mat_from_bits(canvas, 4, i, i);
                    world.h0[i] = h0;
                    world.h1[i] = h0;
                    world.m0[i] = m;
                    world.m1[i] = m;
                }
            }
        }
    }
    world
}

/// Write a playable level directory: output.vmp, world.ini (cloned from the
/// style template with compression disabled), and the template's palette.
pub fn package(world: &World, template_dir: &Path, out_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    world.to_level_data().save_vmp(&out_dir.join("output.vmp"));

    let ini = std::fs::read_to_string(template_dir.join("world.ini"))?;
    let px = (world.width as f32).log2() as u32;
    let py = (world.height as f32).log2() as u32;
    assert_eq!(1usize << px, world.width, "width must be a power of two");
    assert_eq!(1usize << py, world.height, "height must be a power of two");
    let ini: String = ini
        .lines()
        .map(|line| {
            let l = line.trim();
            if l.starts_with("Map Power X") {
                format!("Map Power X={px}")
            } else if l.starts_with("Map Power Y") {
                format!("Map Power Y={py}")
            } else if l.starts_with("Compressed Format Using") {
                "Compressed Format Using=0".to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(out_dir.join("world.ini"), ini)?;

    let ini_text = std::fs::read_to_string(template_dir.join("world.ini"))?;
    let palette_name = ini_text
        .lines()
        .find_map(|l| l.trim().strip_prefix("Palette File="))
        .unwrap_or("harmony.pal")
        .trim()
        .to_string();
    std::fs::copy(
        template_dir.join(&palette_name),
        out_dir.join(&palette_name),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{Crop, Sampler};

    /// pack_full → canvas_to_world must reproduce the source planes.
    #[test]
    fn decode_inverts_packing() {
        let size = 32;
        let total = size * size;
        let mut src = World {
            name: "t".into(),
            width: size,
            height: size,
            h0: (0..total).map(|i| (i * 7 % 250 / 2 * 2) as u8).collect(),
            h1: vec![0; total],
            delta: (0..total).map(|i| ((i / 2) % 16) as u8).collect(),
            dual: (0..total).map(|i| ((i / 2) % 3 == 0) as u8).collect(),
            m0: (0..total).map(|i| ((i / 2) % 8) as u8).collect(),
            m1: (0..total).map(|i| ((i / 2 + 3) % 8) as u8).collect(),
        };
        for i in (0..total).step_by(2) {
            // Pair-constant planes, valid duals, consistent singles.
            src.h0[i + 1] = src.h0[i];
            if src.dual[i] != 0 {
                src.h1[i] = src.h0[i].saturating_add(20);
                src.h1[i + 1] = src.h1[i];
            } else {
                for j in [i, i + 1] {
                    src.h1[j] = src.h0[j];
                    src.delta[j] = 0;
                    src.m1[j] = src.m0[j];
                }
            }
        }
        let worlds = [src];
        let sampler = Sampler::new(&worlds);
        let crop = Crop {
            world: 0,
            x: 0,
            y: 0,
            flip_x: false,
            flip_y: false,
        };
        let mut canvas = Canvas::new(FULL_CHANNELS, size, size);
        sampler.pack_full(&crop, size, &mut canvas.data);
        let back = canvas_to_world(&canvas, "t");
        let src = &worlds[0];
        assert_eq!(src.h0, back.h0);
        assert_eq!(src.h1, back.h1);
        assert_eq!(src.delta, back.delta);
        assert_eq!(src.dual, back.dual);
        assert_eq!(src.m0, back.m0);
        assert_eq!(src.m1, back.m1);
    }
}
