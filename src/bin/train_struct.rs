//! Train the stage-1 structure diffusion U-Net: unconditional-in-space
//! generation of /8-pooled coarse worlds (4 channels) with world-style
//! class conditioning.

use std::path::PathBuf;

use meganeura::Graph;
use van_world::model::{self, UNetConfig};
use van_world::sampler::{COARSE_CHANNELS, Sampler};
use van_world::training::{self, TrainOpts};
use van_world::world::load_all;

struct Args {
    thechain: PathBuf,
    opts: TrainOpts,
    batch: u32,
    resolution: u32,
    base_channels: u32,
    num_levels: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            thechain: "/x/Work/VangersData/thechain".into(),
            opts: TrainOpts {
                out_dir: "checkpoints/struct".into(),
                resume: None,
                steps: 100_000,
                lr: 1e-4,
                save_every: 2000,
                seed: 2,
                duty: 1.0,
            },
            batch: 4,
            resolution: 256,
            base_channels: 64,
            num_levels: 4,
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
                "--levels" => args.num_levels = val().parse().unwrap(),
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
        data_channels: COARSE_CHANNELS as u32,
        cond_channels: 0,
        base_channels: args.base_channels,
        num_levels: args.num_levels,
        ..UNetConfig::sr_default(args.batch, args.resolution)
    };
    let res = args.resolution as usize;
    let data_size = COARSE_CHANNELS * res * res;

    let mut g = Graph::new();
    let (loss, inits) = model::build_training_graph(&mut g, &cfg);
    g.set_outputs(vec![loss]);

    let mut clean = Vec::new();
    let mut labels = Vec::new();
    let batch = args.batch as usize;
    let t_dim = cfg.t_dim;
    training::run(&g, &cfg, &inits, &args.opts, |rng, inp| {
        sampler.fill_batch(rng, batch, res, true, &mut clean, &mut labels);
        for b in 0..batch {
            training::noise_sample(
                rng,
                inp,
                b,
                &clean[b * data_size..(b + 1) * data_size],
                data_size,
                data_size,
                t_dim,
                labels[b],
            );
        }
    });
}
