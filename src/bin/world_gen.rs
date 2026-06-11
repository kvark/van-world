//! End-to-end world generation: stage-1 coarse canvas → ×8 upsample →
//! stage-2 detail canvas → decode → playable level directory.

use std::path::PathBuf;

use meganeura::{Graph, build_inference_session};
use van_world::checkpoint;
use van_world::decode;
use van_world::model::{self, UNetConfig};
use van_world::sampler::{COARSE_CHANNELS, POOL};
use van_world::tiled::{self, TiledOpts};

const WORLD_NAMES: [&str; 10] = [
    "ark-a-znoy",
    "boozeena",
    "fostral",
    "glorx",
    "hmok",
    "khox",
    "necross",
    "threall",
    "weexow",
    "xplo",
];

struct Args {
    thechain: PathBuf,
    struct_ckpt: PathBuf,
    sr_ckpt: PathBuf,
    out_dir: PathBuf,
    class: usize,
    /// World length in full-res texels (power of two).
    length: usize,
    struct_base: u32,
    struct_levels: usize,
    sr_base: u32,
    steps: usize,
    guidance: f32,
    seed: u64,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            thechain: "/x/Work/VangersData/thechain".into(),
            struct_ckpt: "checkpoints/struct/ema.bin".into(),
            sr_ckpt: "checkpoints/sr/ema.bin".into(),
            out_dir: "data/generated/world0".into(),
            class: 2,
            length: 2048,
            struct_base: 64,
            struct_levels: 4,
            sr_base: 64,
            steps: 50,
            guidance: 2.0,
            seed: 11,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().expect("missing value");
            match flag.as_str() {
                "--data" => args.thechain = val().into(),
                "--struct-ckpt" => args.struct_ckpt = val().into(),
                "--sr-ckpt" => args.sr_ckpt = val().into(),
                "--out" => args.out_dir = val().into(),
                "--class" => args.class = val().parse().unwrap(),
                "--length" => args.length = val().parse().unwrap(),
                "--struct-base" => args.struct_base = val().parse().unwrap(),
                "--struct-levels" => args.struct_levels = val().parse().unwrap(),
                "--sr-base" => args.sr_base = val().parse().unwrap(),
                "--steps" => args.steps = val().parse().unwrap(),
                "--guidance" => args.guidance = val().parse().unwrap(),
                "--seed" => args.seed = val().parse().unwrap(),
                other => panic!("unknown flag {other}"),
            }
        }
        assert!(args.length.is_power_of_two());
        args
    }
}

fn load_session(g: Graph, ckpt: &std::path::Path) -> meganeura::runtime::Session {
    let mut session = build_inference_session(&g);
    let (params, step) = checkpoint::load(ckpt).unwrap();
    for (name, data) in &params {
        session.set_parameter(name, data);
    }
    println!("loaded {} (step {step})", ckpt.display());
    session
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    const WIDTH: usize = 2048;

    // --- Stage 1: coarse structure on a (WIDTH/8) × (length/8) torus ---
    let struct_cfg = UNetConfig {
        batch: 2,
        data_channels: COARSE_CHANNELS as u32,
        cond_channels: 0,
        base_channels: args.struct_base,
        num_levels: args.struct_levels,
        ..UNetConfig::sr_default(2, 256)
    };
    let mut g = Graph::new();
    let (pred, _) = model::build_unet(&mut g, &struct_cfg);
    g.set_outputs(vec![pred]);
    let mut session = load_session(g, &args.struct_ckpt);

    let opts = TiledOpts {
        steps: args.steps,
        guidance: args.guidance,
        overlap: 64,
        seed: args.seed,
    };
    println!(
        "stage 1: {}x{} coarse canvas, style {}...",
        WIDTH / POOL,
        args.length / POOL,
        WORLD_NAMES[args.class]
    );
    let coarse = tiled::generate(
        &mut session,
        &struct_cfg,
        WIDTH / POOL,
        args.length / POOL,
        None,
        args.class as u32,
        &opts,
    );
    drop(session);

    // --- Stage 2: ×8 super-resolution ---
    let sr_cfg = UNetConfig {
        batch: 2,
        base_channels: args.sr_base,
        ..UNetConfig::sr_default(2, 128)
    };
    let mut g = Graph::new();
    let (pred, _) = model::build_unet(&mut g, &sr_cfg);
    g.set_outputs(vec![pred]);
    let mut session = load_session(g, &args.sr_ckpt);

    let cond = coarse.upsample(POOL);
    let opts = TiledOpts {
        steps: args.steps,
        guidance: args.guidance,
        overlap: 32,
        seed: args.seed + 1,
    };
    println!("stage 2: {}x{} detail canvas...", WIDTH, args.length);
    let full = tiled::generate(
        &mut session,
        &sr_cfg,
        WIDTH,
        args.length,
        Some(&cond),
        args.class as u32,
        &opts,
    );

    // --- Decode and package ---
    let name = args
        .out_dir
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let world = decode::canvas_to_world(&full, &name);
    let template = args.thechain.join(WORLD_NAMES[args.class]);
    decode::package(&world, &template, &args.out_dir).unwrap();
    println!("world written to {}", args.out_dir.display());
}
