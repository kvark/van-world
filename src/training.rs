//! Stage-agnostic diffusion training loop: session setup, parameter
//! init/resume, Adam, host-side EMA, checkpointing, loss logging.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;

use meganeura::{Graph, build_session};

use crate::checkpoint::{self, Params};
use crate::diffusion;
use crate::model::{Init, Inits, UNetConfig};

pub const NULL_CLASS: u32 = 10;
pub const CLASS_DROPOUT: f64 = 0.1;
const EMA_DECAY: f64 = 0.999;
const EMA_EVERY: u64 = 20;

pub struct TrainOpts {
    pub out_dir: PathBuf,
    pub resume: Option<PathBuf>,
    pub steps: u64,
    pub lr: f32,
    pub save_every: u64,
    pub seed: u64,
}

/// Per-step model inputs filled by the stage-specific closure.
pub struct StepInputs {
    /// [batch * in_channels * res * res], noisy data (+ conditioning).
    pub x: Vec<f32>,
    /// [batch * t_dim]
    pub t_emb: Vec<f32>,
    /// [batch]
    pub classes: Vec<u32>,
    /// [batch * data_channels * res * res]
    pub v_target: Vec<f32>,
}

pub fn init_param(rng: &mut StdRng, size: usize, init: Init) -> Vec<f32> {
    match init {
        Init::Zero => vec![0.0; size],
        Init::One => vec![1.0; size],
        Init::Normal(std) => (0..size)
            .map(|_| rng.sample::<f32, _>(StandardNormal) * std)
            .collect(),
    }
}

/// Noise the data channels of one sample in place and produce its v-target
/// and timestep embedding. Returns nothing; everything written into `inp`.
#[allow(clippy::too_many_arguments)]
pub fn noise_sample(
    rng: &mut StdRng,
    inp: &mut StepInputs,
    sample_index: usize,
    clean: &[f32],
    data_size: usize,
    sample_stride: usize,
    t_dim: usize,
    label: u32,
) {
    let t: f32 = rng.gen_range(0.0..1.0);
    let (alpha, sigma) = diffusion::alpha_sigma(t);
    diffusion::timestep_embedding(
        t,
        t_dim,
        &mut inp.t_emb[sample_index * t_dim..(sample_index + 1) * t_dim],
    );
    inp.classes[sample_index] = if rng.gen_bool(CLASS_DROPOUT) {
        NULL_CLASS
    } else {
        label
    };
    let dst = &mut inp.x[sample_index * sample_stride..];
    for i in 0..data_size {
        let x0 = clean[i];
        let eps: f32 = rng.sample(StandardNormal);
        dst[i] = alpha * x0 + sigma * eps;
        inp.v_target[sample_index * data_size + i] = diffusion::v_target(x0, eps, alpha, sigma);
    }
}

/// Run the training loop. `fill` populates `StepInputs` for each step.
pub fn run(
    g: &Graph,
    cfg: &UNetConfig,
    inits: &Inits,
    opts: &TrainOpts,
    mut fill: impl FnMut(&mut StdRng, &mut StepInputs),
) {
    std::fs::create_dir_all(&opts.out_dir).unwrap();
    let res = cfg.resolution as usize;
    let batch = cfg.batch as usize;
    let data_size = (cfg.data_channels as usize) * res * res;
    let in_size = (cfg.in_channels() as usize) * res * res;

    println!("compiling ({} nodes)...", g.nodes().len());
    let t0 = std::time::Instant::now();
    let mut session = build_session(g);
    println!(
        "compiled in {:.1}s, GPU memory: {}",
        t0.elapsed().as_secs_f64(),
        session.memory_summary()
    );

    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut ema: Params = BTreeMap::new();
    let mut start_step = 0u64;
    if let Some(ref resume) = opts.resume {
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
        let init_map: BTreeMap<&str, Init> = inits.iter().map(|(n, i)| (n.as_str(), *i)).collect();
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

    session.set_adam(opts.lr, 0.9, 0.999, 1e-8);
    session.set_grad_clip_norm(1.0);

    let mut inp = StepInputs {
        x: vec![0.0; batch * in_size],
        t_emb: vec![0.0; batch * cfg.t_dim],
        classes: vec![0; batch],
        v_target: vec![0.0; batch * data_size],
    };
    let mut scratch: Vec<f32> = Vec::new();
    let mut loss_acc = 0.0f64;
    let mut loss_n = 0u64;
    let t_train = std::time::Instant::now();

    for step in start_step..opts.steps {
        fill(&mut rng, &mut inp);

        session.set_input("x", &inp.x);
        session.set_input("t_emb", &inp.t_emb);
        session.set_input_u32("class", &inp.classes);
        session.set_input("v_target", &inp.v_target);
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
        if (step + 1) % opts.save_every == 0 || step + 1 == opts.steps {
            session.wait();
            let mut params = Params::new();
            for name in session
                .param_names()
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
            {
                let size = ema[&name].len();
                let mut data = vec![0.0; size];
                session.read_param(&name, &mut data);
                params.insert(name, data);
            }
            checkpoint::save(&opts.out_dir.join("latest.bin"), &params, step + 1).unwrap();
            checkpoint::save(&opts.out_dir.join("ema.bin"), &ema, step + 1).unwrap();
            println!("checkpoint saved at step {}", step + 1);
        }
    }
    println!("done in {:.1}s", t_train.elapsed().as_secs_f64());
}
