//! Host-side diffusion math: continuous-time cosine schedule with
//! v-prediction (Salimans & Ho 2022) and DDIM sampling.

const COSINE_S: f32 = 0.008;

/// (alpha, sigma) for t in [0, 1]: alpha = sqrt(abar), sigma = sqrt(1 - abar).
pub fn alpha_sigma(t: f32) -> (f32, f32) {
    let f = |u: f32| ((u + COSINE_S) / (1.0 + COSINE_S) * std::f32::consts::FRAC_PI_2).cos();
    let f0 = f(0.0);
    let abar = (f(t) / f0).powi(2).clamp(1e-8, 1.0);
    (abar.sqrt(), (1.0 - abar).sqrt())
}

/// v = alpha * eps - sigma * x0
pub fn v_target(x0: f32, eps: f32, alpha: f32, sigma: f32) -> f32 {
    alpha * eps - sigma * x0
}

/// Recover (x0, eps) estimates from x_t and predicted v.
pub fn from_v(x_t: f32, v: f32, alpha: f32, sigma: f32) -> (f32, f32) {
    (alpha * x_t - sigma * v, sigma * x_t + alpha * v)
}

/// One deterministic DDIM step from t to t_prev, in place over a slice.
pub fn ddim_step(x: &mut [f32], v_pred: &[f32], t: f32, t_prev: f32) {
    let (a, s) = alpha_sigma(t);
    let (ap, sp) = alpha_sigma(t_prev);
    for (x_t, &v) in x.iter_mut().zip(v_pred) {
        let (x0, eps) = from_v(*x_t, v, a, s);
        *x_t = ap * x0 + sp * eps;
    }
}

/// Sinusoidal embedding of t (scaled by 1000), `dim` must be even.
pub fn timestep_embedding(t: f32, dim: usize, out: &mut [f32]) {
    assert_eq!(out.len(), dim);
    let half = dim / 2;
    let scaled = t * 1000.0;
    for i in 0..half {
        let freq = (-(i as f32) * (10000f32).ln() / (half as f32 - 1.0)).exp();
        out[i] = (scaled * freq).sin();
        out[half + i] = (scaled * freq).cos();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_endpoints() {
        let (a0, s0) = alpha_sigma(0.0);
        assert!((a0 - 1.0).abs() < 1e-3 && s0 < 0.05);
        let (a1, s1) = alpha_sigma(1.0);
        assert!(a1 < 0.05 && (s1 - 1.0).abs() < 1e-3);
    }

    #[test]
    fn v_roundtrip() {
        let (a, s) = alpha_sigma(0.37);
        let (x0, eps) = (0.4f32, -1.2f32);
        let x_t = a * x0 + s * eps;
        let v = v_target(x0, eps, a, s);
        let (x0r, epsr) = from_v(x_t, v, a, s);
        assert!((x0 - x0r).abs() < 1e-5 && (eps - epsr).abs() < 1e-5);
    }
}
