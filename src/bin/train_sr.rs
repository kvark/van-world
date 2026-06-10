//! Train the stage-2 super-resolution diffusion U-Net.
//!
//! Each sample is a POOL-aligned full-res crop (10 data channels) paired
//! with its own pooled conditioning (4 channels, replicated to full res).
//! The model learns v-prediction denoising of the data channels given the
//! conditioning channels, a timestep embedding, and a world-style class
//! (dropped to the null class 10% of the time for CFG).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;

use meganeura::{Graph, build_session};
use van_world::checkpoint::{self, Params};
use van_world::diffusion;
use van_world::model::{self, Init, UNetConfig};
use van_world::sampler::{COARSE_CHANNELS, FULL_CHANNELS, Sampler};
use van_world::world::load_all;

const NULL_CLASS: u32 = 10;
const CLASS_DROPOUT: f64 = 0.1;
const EMA_DECAY: f64 = 0.999;
const EMA_EVERY: u64 = 20;

struct Args {
    thechain: PathBuf,
    out_dir: PathBuf,
    resume: Option<PathBuf>,
    batch: u32,
    resolution: u32,
    base_channels: u32,
    steps: u64,
    lr: f32,
    save_every: u64,
    seed: u64,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            thechain: "/x/Work/VangersData/thechain".into(),
            out_dir: "checkpoints/sr".into(),
            resume: None,
            batch: 4,
            resolution: 128,
            base_channels: 64,
            steps: 100_000,
            lr: 1e-4,
            save_every: 2000,
            seed: 1,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().expect("missing value");
            match flag.as_str() {
                "--data" => args.thechain = val().into(),
                "--out" => args.out_dir = val().into(),
                "--resume" => args.resume = Some(val().into()),
                "--batch" => args.batch = val().parse().unwrap(),
                "--res" => args.resolution = val().parse().unwrap(),
                "--base" => args.base_channels = val().parse().unwrap(),
                "--steps" => args.steps = val().parse().unwrap(),
                "--lr" => args.lr = val().parse().unwrap(),
                "--save-every" => args.save_every = val().parse().unwrap(),
                "--seed" => args.seed = val().parse().unwrap(),
                other => panic!("unknown flag {other}"),
            }
        }
        args
    }
}

fn init_param(rng: &mut StdRng, size: usize, init: Init) -> Vec<f32> {
    match init {
        Init::Zero => vec![0.0; size],
        Init::One => vec![1.0; size],
        Init::Normal(std) => (0..size)
            .map(|_| rng.sample::<f32, _>(StandardNormal) * std)
            .collect(),
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    std::fs::create_dir_all(&args.out_dir).unwrap();

    println!("loading worlds from {}...", args.thechain.display());
    let worlds = load_all(&args.thechain);
    assert_eq!(worlds.len(), NULL_CLASS as usize, "class count mismatch");
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

    println!("building graph...");
    let mut g = Graph::new();
    let (loss, inits) = model::build_training_graph(&mut g, &cfg);
    g.set_outputs(vec![loss]);
    println!("compiling ({} nodes)...", g.nodes().len());
    let t0 = std::time::Instant::now();
    let mut session = build_session(&g);
    println!(
        "compiled in {:.1}s, GPU memory: {}",
        t0.elapsed().as_secs_f64(),
        session.memory_summary()
    );

    // --- Parameter init / resume ---
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut ema: Params = BTreeMap::new();
    let mut start_step = 0u64;
    if let Some(ref resume) = args.resume {
        let (params, step) = checkpoint::load(resume).unwrap();
        let (ema_params, _) =
            checkpoint::load(&resume.with_file_name("ema.bin")).unwrap_or((params.clone(), 0));
        for (name, data) in &params {
            session.set_parameter(name, data);
        }
        ema = ema_params;
        start_step = step;
        session.set_adam_step_count(step as u32);
        println!("resumed from {} at step {step}", resume.display());
    } else {
        let init_map: BTreeMap<&str, Init> =
            inits.iter().map(|(n, i)| (n.as_str(), *i)).collect();
        let mut num_params = 0usize;
        for (name, buf) in session.plan().param_buffers.clone() {
            let size = session.plan().buffers[buf.0 as usize] / 4;
            let init = *init_map
                .get(name.as_str())
                .unwrap_or_else(|| panic!("no init recorded for {name}"));
            let data = init_param(&mut rng, size, init);
            session.set_parameter(&name, &data);
            ema.insert(name.clone(), data);
            num_params += size;
        }
        println!(
            "initialized {num_params} parameters ({:.1} MB)",
            num_params as f64 * 4.0 / 1e6
        );
    }

    session.set_adam(args.lr, 0.9, 0.999, 1e-8);
    session.set_grad_clip_norm(1.0);

    // --- Training loop ---
    let mut pairs = Vec::new();
    let mut labels = Vec::new();
    let mut x_input = vec![0.0f32; args.batch as usize * pair_size];
    let mut v_target = vec![0.0f32; args.batch as usize * data_size];
    let mut t_emb = vec![0.0f32; args.batch as usize * cfg.t_dim];
    let mut classes = vec![0u32; args.batch as usize];
    let mut scratch = vec![0.0f32; 0];

    let mut loss_acc = 0.0f64;
    let mut loss_n = 0u64;
    let t_train = std::time::Instant::now();

    for step in start_step..args.steps {
        sampler.fill_batch_pair(&mut rng, args.batch as usize, res, &mut pairs, &mut labels);

        for b in 0..args.batch as usize {
            let src = &pairs[b * pair_size..(b + 1) * pair_size];
            let t: f32 = rng.gen_range(0.0..1.0);
            let (alpha, sigma) = diffusion::alpha_sigma(t);
            diffusion::timestep_embedding(t, cfg.t_dim, &mut t_emb[b * cfg.t_dim..(b + 1) * cfg.t_dim]);
            classes[b] = if rng.gen_bool(CLASS_DROPOUT) {
                NULL_CLASS
            } else {
                labels[b]
            };

            let dst = &mut x_input[b * pair_size..(b + 1) * pair_size];
            // Noise the data channels; v-prediction target.
            for i in 0..data_size {
                let x0 = src[i];
                let eps: f32 = rng.sample(StandardNormal);
                dst[i] = alpha * x0 + sigma * eps;
                v_target[b * data_size + i] = diffusion::v_target(x0, eps, alpha, sigma);
            }
            // Conditioning augmentation: mild noise on the coarse channels.
            let aug = rng.gen_range(0.0..0.1);
            for i in data_size..pair_size {
                let eps: f32 = rng.sample(StandardNormal);
                dst[i] = src[i] + aug * eps;
            }
        }

        session.set_input("x", &x_input);
        session.set_input("t_emb", &t_emb);
        session.set_input_u32("class", &classes);
        session.set_input("v_target", &v_target);
        session.step();

        if (step + 1) % EMA_EVERY == 0 {
            session.wait();
            let decay = EMA_DECAY.powi(EMA_EVERY as i32) as f32;
            for (name, ema_data) in ema.iter_mut() {
                scratch.resize(ema_data.len(), 0.0);
                session.read_param(name, &mut scratch);
                for (e, &p) in ema_data.iter_mut().zip(&scratch) {
                    *e = decay * *e + (1.0 - decay) * p;
                }
            }
        }

        if (step + 1) % 50 == 0 {
            session.wait();
            loss_acc += session.read_loss() as f64;
            loss_n += 1;
        }
        if (step + 1) % 500 == 0 {
            let sps = (step + 1 - start_step) as f64 / t_train.elapsed().as_secs_f64();
            println!(
                "step {:>7}  loss {:.5}  ({:.2} steps/s)",
                step + 1,
                loss_acc / loss_n.max(1) as f64,
                sps
            );
            loss_acc = 0.0;
            loss_n = 0;
        }
        if (step + 1) % args.save_every == 0 || step + 1 == args.steps {
            session.wait();
            let mut params = Params::new();
            for name in session.param_names().into_iter().map(String::from).collect::<Vec<_>>() {
                let size = ema[&name].len();
                let mut data = vec![0.0; size];
                session.read_param(&name, &mut data);
                params.insert(name, data);
            }
            checkpoint::save(&args.out_dir.join("latest.bin"), &params, step + 1).unwrap();
            checkpoint::save(&args.out_dir.join("ema.bin"), &ema, step + 1).unwrap();
            println!("checkpoint saved at step {}", step + 1);
        }
    }
    println!("done in {:.1}s", t_train.elapsed().as_secs_f64());
}

#[allow(dead_code)]
fn unused(_: &Path) {}
