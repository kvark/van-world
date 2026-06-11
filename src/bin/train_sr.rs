//! Train the stage-2 super-resolution diffusion U-Net.
//!
//! Each sample is a POOL-aligned full-res crop (10 data channels) paired
//! with its own pooled conditioning (4 channels, replicated to full res).
//! The model learns v-prediction denoising of the data channels given the
//! conditioning channels, a timestep embedding, and a world-style class
//! (dropped to the null class for CFG).

use std::path::PathBuf;

use meganeura::Graph;
use rand::Rng;
use rand_distr::StandardNormal;
use van_world::model::{self, UNetConfig};
use van_world::sampler::{COARSE_CHANNELS, FULL_CHANNELS, Sampler};
use van_world::training::{self, TrainOpts};
use van_world::world::load_all;

struct Args {
    thechain: PathBuf,
    opts: TrainOpts,
    batch: u32,
    resolution: u32,
    base_channels: u32,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            thechain: "/x/Work/VangersData/thechain".into(),
            opts: TrainOpts {
                out_dir: "checkpoints/sr".into(),
                resume: None,
                steps: 100_000,
                lr: 1e-4,
                save_every: 2000,
                seed: 1,
                duty: 1.0,
            },
            batch: 4,
            resolution: 128,
            base_channels: 64,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().expect("missing value");
            match flag.as_str() {
                "--data" => args.thechain = val().into(),
                "--out" => args.opts.out_dir = val().into(),
                "--resume" => args.opts.resume = Some(val().into()),
                "--batch" => args.batch = val().parse().unwrap(),
                "--res" => args.resolution = val().parse().unwrap(),
                "--base" => args.base_channels = val().parse().unwrap(),
                "--steps" => args.opts.steps = val().parse().unwrap(),
                "--lr" => args.opts.lr = val().parse().unwrap(),
                "--save-every" => args.opts.save_every = val().parse().unwrap(),
                "--seed" => args.opts.seed = val().parse().unwrap(),
                "--duty" => args.opts.duty = val().parse().unwrap(),
                other => panic!("unknown flag {other}"),
            }
        }
        args
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    println!("loading worlds from {}...", args.thechain.display());
    let worlds = load_all(&args.thechain);
    let sampler = Sampler::new(&worlds);

    let cfg = UNetConfig {
        batch: args.batch,
        base_channels: args.base_channels,
        ..UNetConfig::sr_default(args.batch, args.resolution)
    };
    let res = args.resolution as usize;
    let plane = res * res;
    let data_size = FULL_CHANNELS * plane;
    let pair_size = (FULL_CHANNELS + COARSE_CHANNELS) * plane;

    let mut g = Graph::new();
    let (loss, inits) = model::build_training_graph(&mut g, &cfg);
    g.set_outputs(vec![loss]);

    let mut pairs = Vec::new();
    let mut labels = Vec::new();
    let batch = args.batch as usize;
    let t_dim = cfg.t_dim;
    training::run(&g, &cfg, &inits, &args.opts, |rng, inp| {
        sampler.fill_batch_pair(rng, batch, res, &mut pairs, &mut labels);
        for b in 0..batch {
            let src = &pairs[b * pair_size..(b + 1) * pair_size];
            training::noise_sample(rng, inp, b, &src[..data_size], data_size, pair_size, t_dim, labels[b]);
            // Conditioning channels pass through clean, with mild noise
            // augmentation to match imperfect stage-1 outputs at inference.
            let aug = rng.gen_range(0.0..0.1);
            let dst = &mut inp.x[b * pair_size..(b + 1) * pair_size];
            for i in data_size..pair_size {
                let eps: f32 = rng.sample(StandardNormal);
                dst[i] = src[i] + aug * eps;
            }
        }
    });
}
