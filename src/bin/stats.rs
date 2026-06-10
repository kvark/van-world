use std::io::Write as _;
use std::path::Path;

use rand::{SeedableRng, rngs::StdRng};
use van_world::sampler::{COARSE_CHANNELS, FULL_CHANNELS, POOL, Sampler};
use van_world::world::{NUM_TERRAINS, load_all};

fn write_pgm(path: &Path, size: usize, plane: &[f32]) {
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "P5\n{} {}\n255\n", size, size).unwrap();
    let bytes: Vec<u8> = plane
        .iter()
        .map(|&v| ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
        .collect();
    f.write_all(&bytes).unwrap();
}

fn main() {
    let thechain = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/x/Work/VangersData/thechain".to_string());
    let worlds = load_all(Path::new(&thechain));

    println!(
        "{:12} {:>5}x{:<5} {:>6} {:>7} {:>7}  terrain histogram (%)",
        "world", "w", "h", "dual%", "h0 avg", "d>0%"
    );
    for w in &worlds {
        let total = (w.width * w.height) as f64;
        let dual: u64 = w.dual.iter().map(|&d| d as u64).sum();
        let h_avg: f64 = w.h0.iter().map(|&h| h as f64).sum::<f64>() / total;
        let dual_zero_delta = w
            .dual
            .iter()
            .zip(&w.delta)
            .filter(|&(&d, &dl)| d != 0 && dl == 0)
            .count();
        let mut hist = [0u64; NUM_TERRAINS as usize];
        for &m in &w.m0 {
            hist[m as usize] += 1;
        }
        let hist_pct: Vec<String> = hist
            .iter()
            .map(|&c| format!("{:.1}", c as f64 / total * 100.0))
            .collect();
        println!(
            "{:12} {:>5}x{:<5} {:>5.1}% {:>7.1} {:>6.1}%  [{}]",
            w.name,
            w.width,
            w.height,
            dual as f64 / total * 100.0,
            h_avg,
            dual_zero_delta as f64 / total * 100.0,
            hist_pct.join(" ")
        );
    }

    // Dump a few sample crops for eyeballing.
    let out_dir = Path::new("data/preview");
    std::fs::create_dir_all(out_dir).unwrap();
    let sampler = Sampler::new(&worlds);
    let mut rng = StdRng::seed_from_u64(42);
    let size = 256;
    for i in 0..4 {
        let crop = sampler.sample_crop(&mut rng);
        let mut full = vec![0.0; FULL_CHANNELS * size * size];
        sampler.pack_full(&crop, size, &mut full);
        let plane = size * size;
        write_pgm(
            &out_dir.join(format!("crop{i}_{}_h0.pgm", worlds[crop.world].name)),
            size,
            &full[..plane],
        );
        write_pgm(
            &out_dir.join(format!("crop{i}_{}_dual.pgm", worlds[crop.world].name)),
            size,
            &full[3 * plane..4 * plane],
        );

        let csize = size / POOL;
        let mut coarse = vec![0.0; COARSE_CHANNELS * csize * csize];
        sampler.pack_coarse(&crop, csize, &mut coarse);
        write_pgm(
            &out_dir.join(format!("crop{i}_{}_coarse_h0.pgm", worlds[crop.world].name)),
            csize,
            &coarse[..csize * csize],
        );
    }
    println!("previews written to {}", out_dir.display());
}
