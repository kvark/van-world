//! Conditional diffusion U-Net built on meganeura.
//!
//! Extends meganeura's reference SD U-Net with timestep + class
//! conditioning. Per-(batch, channel) FiLM shifts are materialized with
//! differentiable ops only: project the embedding to [B*C, 1] and matmul
//! against a constant ones row [1, H*W].
//!
//! Graph inputs:
//! - "x":       [batch * (data_channels + cond_channels) * res * res]
//! - "t_emb":   [batch, t_dim]  (host-computed sinusoidal embedding)
//! - "class":   [batch] u32     (world style; last index = null for CFG)
//! - "v_target" [batch * data_channels * res * res]  (training graph only)

use meganeura::graph::{Graph, NodeId};
use meganeura::nn::Linear;

pub struct UNetConfig {
    pub batch: u32,
    pub data_channels: u32,
    pub cond_channels: u32,
    pub base_channels: u32,
    pub num_levels: usize,
    pub resolution: u32,
    pub num_groups: u32,
    pub gn_eps: f32,
    pub t_dim: usize,
    pub emb_dim: usize,
    /// Including the trailing null class used for CFG dropout.
    pub num_classes: usize,
}

impl UNetConfig {
    pub fn sr_default(batch: u32, resolution: u32) -> Self {
        Self {
            batch,
            data_channels: crate::sampler::FULL_CHANNELS as u32,
            cond_channels: crate::sampler::COARSE_CHANNELS as u32,
            base_channels: 64,
            num_levels: 3,
            resolution,
            num_groups: 16,
            gn_eps: 1e-5,
            t_dim: 128,
            emb_dim: 256,
            num_classes: 11,
        }
    }

    pub fn in_channels(&self) -> u32 {
        self.data_channels + self.cond_channels
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Init {
    Zero,
    One,
    /// Normal with the given std deviation.
    Normal(f32),
}

/// Parameter initializers recorded during graph construction.
pub type Inits = Vec<(String, Init)>;

struct Ctx<'a> {
    g: &'a mut Graph,
    inits: Inits,
}

impl Ctx<'_> {
    fn param(&mut self, name: &str, shape: &[usize], init: Init) -> NodeId {
        self.inits.push((name.to_string(), init));
        self.g.parameter(name, shape)
    }

    fn linear(&mut self, name: &str, in_f: usize, out_f: usize, init: Init) -> Linear {
        self.inits.push((format!("{name}.weight"), init));
        self.inits.push((format!("{name}.bias"), Init::Zero));
        Linear::new(self.g, name, in_f, out_f)
    }

    fn conv(&mut self, name: &str, in_c: u32, out_c: u32, k: u32, zero: bool) -> NodeId {
        let fan_in = (in_c * k * k) as f32;
        let init = if zero {
            Init::Zero
        } else {
            Init::Normal((2.0 / fan_in).sqrt())
        };
        self.param(name, &[(out_c * in_c * k * k) as usize], init)
    }
}

struct Spatial {
    h: u32,
    w: u32,
    c: u32,
}

fn xavier_std(in_f: usize, out_f: usize) -> Init {
    Init::Normal((2.0 / (in_f + out_f) as f32).sqrt())
}

/// FiLM shift: emb [B, emb_dim] -> [B, out_c] -> broadcast plane added to x.
fn film_shift(
    ctx: &mut Ctx,
    x: NodeId,
    emb: NodeId,
    prefix: &str,
    cfg: &UNetConfig,
    s: &Spatial,
) -> NodeId {
    let out_c = s.c;
    // Zero-init so training starts as the unconditional model.
    let proj = ctx.linear(
        &format!("{prefix}.emb_proj"),
        cfg.emb_dim,
        out_c as usize,
        Init::Zero,
    );
    let e = proj.forward(ctx.g, emb); // [B, out_c]
    let col = ctx
        .g
        .reshape(e, &[(cfg.batch * out_c) as usize, 1]);
    let spatial = (s.h * s.w) as usize;
    let ones = ctx.g.constant(vec![1.0; spatial], &[1, spatial]);
    let plane = ctx.g.matmul(col, ones); // [B*out_c, spatial]
    let flat = ctx
        .g
        .reshape(plane, &[(cfg.batch * out_c) as usize * spatial]);
    ctx.g.add(x, flat)
}

fn resblock(
    ctx: &mut Ctx,
    x: NodeId,
    emb: NodeId,
    prefix: &str,
    cfg: &UNetConfig,
    s: &Spatial,
    out_c: u32,
) -> NodeId {
    let batch = cfg.batch;
    let spatial = s.h * s.w;
    let in_c = s.c;

    let gn1_w = ctx.param(&format!("{prefix}.norm1.weight"), &[in_c as usize], Init::One);
    let gn1_b = ctx.param(&format!("{prefix}.norm1.bias"), &[in_c as usize], Init::Zero);
    let h = ctx
        .g
        .group_norm(x, gn1_w, gn1_b, batch, in_c, spatial, cfg.num_groups, cfg.gn_eps);
    let h = ctx.g.silu(h);
    let conv1_w = ctx.conv(&format!("{prefix}.conv1.weight"), in_c, out_c, 3, false);
    let h = ctx
        .g
        .conv2d(h, conv1_w, batch, in_c, s.h, s.w, out_c, 3, 3, 1, 1);

    let mid_s = Spatial { h: s.h, w: s.w, c: out_c };
    let h = film_shift(ctx, h, emb, prefix, cfg, &mid_s);

    let gn2_w = ctx.param(&format!("{prefix}.norm2.weight"), &[out_c as usize], Init::One);
    let gn2_b = ctx.param(&format!("{prefix}.norm2.bias"), &[out_c as usize], Init::Zero);
    let h = ctx
        .g
        .group_norm(h, gn2_w, gn2_b, batch, out_c, spatial, cfg.num_groups, cfg.gn_eps);
    let h = ctx.g.silu(h);
    let conv2_w = ctx.conv(&format!("{prefix}.conv2.weight"), out_c, out_c, 3, false);
    let h = ctx
        .g
        .conv2d(h, conv2_w, batch, out_c, s.h, s.w, out_c, 3, 3, 1, 1);

    if in_c == out_c {
        ctx.g.add(x, h)
    } else {
        let res_w = ctx.conv(&format!("{prefix}.res_conv.weight"), in_c, out_c, 1, false);
        let x_proj = ctx
            .g
            .conv2d(x, res_w, batch, in_c, s.h, s.w, out_c, 1, 1, 1, 0);
        ctx.g.add(x_proj, h)
    }
}

/// Build the forward pass; returns (prediction node, parameter inits).
pub fn build_unet(g: &mut Graph, cfg: &UNetConfig) -> (NodeId, Inits) {
    let mut ctx = Ctx { g, inits: Vec::new() };
    let ctx = &mut ctx;

    let batch = cfg.batch;
    let res = cfg.resolution;
    let in_c = cfg.in_channels();
    let in_size = (batch * in_c * res * res) as usize;

    let x_in = ctx.g.input("x", &[in_size]);

    // --- Conditioning trunk: t_emb + class embedding -> emb [B, emb_dim] ---
    let t_in = ctx.g.input("t_emb", &[batch as usize, cfg.t_dim]);
    let t_proj = ctx.linear("cond.t_proj", cfg.t_dim, cfg.emb_dim, xavier_std(cfg.t_dim, cfg.emb_dim));
    let t_emb = t_proj.forward(ctx.g, t_in);

    let class_in = ctx.g.input_u32("class", &[batch as usize]);
    let class_table = ctx.param(
        "cond.class_table",
        &[cfg.num_classes, cfg.emb_dim],
        Init::Normal(0.02),
    );
    let c_emb = ctx.g.embedding(class_in, class_table);

    let emb = ctx.g.add(t_emb, c_emb);
    let emb = ctx.g.silu(emb);
    let trunk = ctx.linear("cond.trunk", cfg.emb_dim, cfg.emb_dim, xavier_std(cfg.emb_dim, cfg.emb_dim));
    let emb = trunk.forward(ctx.g, emb);
    let emb = ctx.g.silu(emb);

    // --- U-Net ---
    let base_c = cfg.base_channels;
    let conv_in_w = ctx.conv("conv_in.weight", in_c, base_c, 3, false);
    let mut x = ctx
        .g
        .conv2d(x_in, conv_in_w, batch, in_c, res, res, base_c, 3, 3, 1, 1);
    let mut s = Spatial { h: res, w: res, c: base_c };

    let ch_mults: Vec<u32> = (0..cfg.num_levels).map(|i| 1u32 << i).collect();
    let mut skips: Vec<(NodeId, Spatial)> = Vec::new();

    for (level, &mult) in ch_mults.iter().enumerate() {
        let out_c = base_c * mult;
        x = resblock(ctx, x, emb, &format!("encoder.{level}.resblock"), cfg, &s, out_c);
        s.c = out_c;
        skips.push((x, Spatial { h: s.h, w: s.w, c: s.c }));
        if level < cfg.num_levels - 1 {
            let down_w = ctx.conv(&format!("encoder.{level}.downsample.weight"), out_c, out_c, 3, false);
            x = ctx
                .g
                .conv2d(x, down_w, batch, out_c, s.h, s.w, out_c, 3, 3, 2, 1);
            s.h = (s.h + 2 - 3) / 2 + 1;
            s.w = (s.w + 2 - 3) / 2 + 1;
        }
    }

    x = resblock(ctx, x, emb, "middle.resblock", cfg, &s, s.c);

    for level in (0..cfg.num_levels).rev() {
        let out_c = base_c * ch_mults[level];
        if level < cfg.num_levels - 1 {
            x = ctx.g.upsample_2x(x, batch, s.c, s.h, s.w);
            s.h *= 2;
            s.w *= 2;
        }
        let &(skip, ref skip_s) = &skips[level];
        assert_eq!((s.h, s.w), (skip_s.h, skip_s.w));
        x = ctx.g.concat(x, skip, batch, s.c, skip_s.c, s.h * s.w);
        let dec_s = Spatial { h: s.h, w: s.w, c: s.c + skip_s.c };
        x = resblock(ctx, x, emb, &format!("decoder.{level}.resblock"), cfg, &dec_s, out_c);
        s.c = out_c;
    }

    let gn_w = ctx.param("conv_out.norm.weight", &[base_c as usize], Init::One);
    let gn_b = ctx.param("conv_out.norm.bias", &[base_c as usize], Init::Zero);
    x = ctx
        .g
        .group_norm(x, gn_w, gn_b, batch, base_c, res * res, cfg.num_groups, cfg.gn_eps);
    x = ctx.g.silu(x);
    let conv_out_w = ctx.conv("conv_out.weight", base_c, cfg.data_channels, 3, true);
    let pred = ctx.g.conv2d(
        x,
        conv_out_w,
        batch,
        base_c,
        res,
        res,
        cfg.data_channels,
        3,
        3,
        1,
        1,
    );
    (pred, std::mem::take(&mut ctx.inits))
}

/// Build the training graph; returns (loss node, parameter inits).
pub fn build_training_graph(g: &mut Graph, cfg: &UNetConfig) -> (NodeId, Inits) {
    let (pred, inits) = build_unet(g, cfg);
    let out_size = (cfg.batch * cfg.data_channels * cfg.resolution * cfg.resolution) as usize;
    let target = g.input("v_target", &[out_size]);
    (g.mse_loss(pred, target), inits)
}
