//! Super-resolve a real world's coarse representation with the trained
//! stage-2 model and write previews next to the ground truth.
//!
//! Runs the U-Net with batch 2: sample 0 conditioned on the world class,
//! sample 1 on the null class, combined per DDIM step via classifier-free
//! guidance.

use std::path::PathBuf;

use meganeura::{Graph, build_inference_session};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use van_world::checkpoint;
use van_world::diffusion;
use van_world::model::{self, UNetConfig};
use van_world::sampler::{COARSE_CHANNELS, FULL_CHANNELS, POOL, Crop, Sampler};
use van_world::world::load_all;

const NULL_CLASS: u32 = 10;

struct Args {
    thechain: PathBuf,
    ckpt: PathBuf,
    out_dir: PathBuf,
    world: usize,
    resolution: u32,
    base_channels: u32,
    steps: usize,
    guidance: f32,
    seed: u64,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            thechain: "/x/Work/VangersData/thechain".into(),
            ckpt: "checkpoints/sr/ema.bin".into(),
            out_dir: "data/sr_samples".into(),
            world: 2, // fostral
            resolution: 128,
            base_channels: 64,
            steps: 50,
            guidance: 2.0,
            seed: 7,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().expect("missing value");
            match flag.as_str() {
                "--data" => args.thechain = val().into(),
                "--ckpt" => args.ckpt = val().into(),
                "--out" => args.out_dir = val().into(),
                "--world" => args.world = val().parse().unwrap(),
                "--res" => args.resolution = val().parse().unwrap(),
                "--base" => args.base_channels = val().parse().unwrap(),
                "--steps" => args.steps = val().parse().unwrap(),
                "--guidance" => args.guidance = val().parse().unwrap(),
                "--seed" => args.seed = val().parse().unwrap(),
                other => panic!("unknown flag {other}"),
            }
        }
        args
    }
}

fn write_pgm(path: &std::path::Path, size: usize, plane: &[f32]) {
    use std::io::Write as _;
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "P5\n{} {}\n255\n", size, size).unwrap();
    let bytes: Vec<u8> = plane
        .iter()
        .map(|&v| ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
        .collect();
    f.write_all(&bytes).unwrap();
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.out_dir).unwrap();

    let worlds = load_all(&args.thechain);
    let sampler = Sampler::new(&worlds);
    let world = &worlds[args.world];
    println!("conditioning on {} (class {})", world.name, args.world);

    let cfg = UNetConfig {
        batch: 2,
        base_channels: args.base_channels,
        ..UNetConfig::sr_default(2, args.resolution)
    };
    let res = args.resolution as usize;
    let plane = res * res;
    let data_size = FULL_CHANNELS * plane;
    let pair_size = (FULL_CHANNELS + COARSE_CHANNELS) * plane;

    let mut g = Graph::new();
    let (pred, _inits) = model::build_unet(&mut g, &cfg);
    g.set_outputs(vec![pred]);
    let mut session = build_inference_session(&g);

    let (params, step) = checkpoint::load(&args.ckpt).unwrap();
    for (name, data) in &params {
        session.set_parameter(name, data);
    }
    println!("loaded {} (step {step})", args.ckpt.display());

    // Ground-truth pair from a POOL-aligned crop of the chosen world.
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut crop = Crop {
        world: args.world,
        x: rng.gen_range(0..world.width),
        y: rng.gen_range(0..world.height),
        flip_x: false,
        flip_y: false,
    };
    crop.x &= !(POOL - 1);
    crop.y &= !(POOL - 1);
    let mut truth = vec![0.0f32; pair_size];
    sampler.pack_pair(&crop, res, &mut truth);
    let cond = &truth[data_size..];

    // DDIM loop.
    let mut x: Vec<f32> = (0..data_size)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    let mut x_input = vec![0.0f32; 2 * pair_size];
    let mut t_emb = vec![0.0f32; 2 * cfg.t_dim];
    let classes = [args.world as u32, NULL_CLASS];
    let mut v_guided = vec![0.0f32; data_size];

    let t_start = std::time::Instant::now();
    for i in 0..args.steps {
        let t = 1.0 - i as f32 / args.steps as f32;
        let t_prev = 1.0 - (i + 1) as f32 / args.steps as f32;
        for b in 0..2 {
            x_input[b * pair_size..b * pair_size + data_size].copy_from_slice(&x);
            x_input[b * pair_size + data_size..(b + 1) * pair_size].copy_from_slice(cond);
            diffusion::timestep_embedding(t, cfg.t_dim, &mut t_emb[b * cfg.t_dim..(b + 1) * cfg.t_dim]);
        }
        session.set_input("x", &x_input);
        session.set_input("t_emb", &t_emb);
        session.set_input_u32("class", &classes);
        session.step();
        session.wait();
        let v = session.read_output(2 * data_size);
        for j in 0..data_size {
            v_guided[j] = v[data_size + j] + args.guidance * (v[j] - v[data_size + j]);
        }
        diffusion::ddim_step(&mut x, &v_guided, t, t_prev);
    }
    println!(
        "sampled {} steps in {:.1}s",
        args.steps,
        t_start.elapsed().as_secs_f64()
    );

    let tag = format!("{}_{}_{}", world.name, crop.x, crop.y);
    write_pgm(&args.out_dir.join(format!("{tag}_h0_gen.pgm")), res, &x[..plane]);
    write_pgm(&args.out_dir.join(format!("{tag}_h0_real.pgm")), res, &truth[..plane]);
    write_pgm(&args.out_dir.join(format!("{tag}_h0_coarse.pgm")), res, &truth[data_size..data_size + plane]);
    write_pgm(&args.out_dir.join(format!("{tag}_dual_gen.pgm")), res, &x[3 * plane..4 * plane]);
    write_pgm(&args.out_dir.join(format!("{tag}_dual_real.pgm")), res, &truth[3 * plane..4 * plane]);
    println!("previews written to {}/{tag}_*.pgm", args.out_dir.display());
}
