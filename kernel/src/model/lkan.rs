use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{Init, VarBuilder, init, ops};

#[derive(Debug, Clone)]
pub struct LiquidKanConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub cheb_order: usize,
    pub dt: f64,
    pub tau_min: f64,
    pub x_scale: f64,
}

impl Default for LiquidKanConfig {
    fn default() -> Self {
        Self {
            in_dim: 1,
            hidden_dim: 8,
            out_dim: 1,
            cheb_order: 5,
            dt: 0.05,
            tau_min: 1e-3,
            x_scale: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LiquidKanLayer {
    cfg: LiquidKanConfig,
    w_base: Tensor,  // [in_dim, hidden_dim]
    w_cheb: Tensor,  // [in_dim, cheb_order + 1, hidden_dim]
    b_phi: Tensor,   // [hidden_dim]
    tau_raw: Tensor, // [hidden_dim]
    w_out: Tensor,   // [hidden_dim, out_dim]
    b_out: Tensor,   // [out_dim]
}

impl LiquidKanLayer {
    pub fn new(vb: VarBuilder<'_>, cfg: LiquidKanConfig) -> Result<Self> {
        if cfg.in_dim == 0 || cfg.hidden_dim == 0 || cfg.out_dim == 0 {
            bail!("all dimensions must be > 0, got config: {cfg:?}");
        }
        if cfg.dt <= 0.0 {
            bail!("dt must be > 0, got {}", cfg.dt);
        }
        if cfg.tau_min <= 0.0 {
            bail!("tau_min must be > 0, got {}", cfg.tau_min);
        }
        if cfg.x_scale <= 0.0 {
            bail!("x_scale must be > 0, got {}", cfg.x_scale);
        }

        let vb = vb.pp("liquid_kan");

        let w_base = vb
            .get_with_hints(
                (cfg.in_dim, cfg.hidden_dim),
                "w_base",
                init::DEFAULT_KAIMING_NORMAL,
            )
            .context("failed to init w_base")?;
        let w_cheb = vb
            .get_with_hints(
                (cfg.in_dim, cfg.cheb_order + 1, cfg.hidden_dim),
                "w_cheb",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .context("failed to init w_cheb")?;
        let b_phi = vb
            .get_with_hints(cfg.hidden_dim, "b_phi", Init::Const(0.0))
            .context("failed to init b_phi")?;
        let tau_raw = vb
            .get_with_hints(cfg.hidden_dim, "tau_raw", Init::Const(0.5))
            .context("failed to init tau_raw")?;
        let w_out = vb
            .get_with_hints(
                (cfg.hidden_dim, cfg.out_dim),
                "w_out",
                init::DEFAULT_KAIMING_NORMAL,
            )
            .context("failed to init w_out")?;
        let b_out = vb
            .get_with_hints(cfg.out_dim, "b_out", Init::Const(0.0))
            .context("failed to init b_out")?;

        Ok(Self {
            cfg,
            w_base,
            w_cheb,
            b_phi,
            tau_raw,
            w_out,
            b_out,
        })
    }

    pub fn config(&self) -> &LiquidKanConfig {
        &self.cfg
    }

    pub fn zero_state(&self, batch_size: usize) -> Result<Tensor> {
        Tensor::zeros(
            (batch_size, self.cfg.hidden_dim),
            self.w_base.dtype(),
            self.w_base.device(),
        )
        .context("failed to create zero liquid state")
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        // Clamp before exp to avoid overflow in pure tensor ops.
        let x = x.clamp(-20.0, 20.0)?;
        ((x.exp()? + 1.0)?).log().context("softplus failed")
    }

    fn tau(&self) -> Result<Tensor> {
        // tau > 0 via softplus(raw) + tau_min.
        (self.stable_softplus(&self.tau_raw)? + self.cfg.tau_min).context("tau computation failed")
    }

    fn chebyshev_basis(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, in_dim) = x
            .dims2()
            .context("expected x to be rank-2 [batch, in_dim] in chebyshev_basis")?;
        if in_dim != self.cfg.in_dim {
            bail!(
                "x in_dim mismatch in chebyshev_basis: expected {}, got {}",
                self.cfg.in_dim,
                in_dim
            );
        }

        let order = self.cfg.cheb_order;
        let x_scaled = ((x / self.cfg.x_scale)?).clamp(-1.0, 1.0)?;
        let mut basis: Vec<Tensor> = Vec::with_capacity(order + 1);

        let t0 = x_scaled.ones_like()?;
        basis.push(t0.clone());

        if order >= 1 {
            let mut t_prev = t0;
            let mut t_curr = x_scaled.clone();
            basis.push(t_curr.clone());

            for _ in 2..=order {
                let t_next = (((&x_scaled * &t_curr)? * 2.0)? - &t_prev)?;
                basis.push(t_next.clone());
                t_prev = t_curr;
                t_curr = t_next;
            }
        }

        let refs: Vec<&Tensor> = basis.iter().collect();
        Tensor::stack(&refs, 2).context("failed to stack Chebyshev basis")
    }

    fn phi(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, in_dim) = x
            .dims2()
            .context("expected x to be rank-2 [batch, in_dim] in phi")?;
        if in_dim != self.cfg.in_dim {
            bail!(
                "x in_dim mismatch in phi: expected {}, got {}",
                self.cfg.in_dim,
                in_dim
            );
        }

        let base_term = ops::silu(x)?
            .matmul(&self.w_base)
            .context("base term matmul failed")?;

        let cheb_basis = self.chebyshev_basis(x)?;
        let cheb_basis = cheb_basis
            .flatten_from(1)
            .context("failed to flatten Chebyshev basis")?;
        let cheb_weights = self
            .w_cheb
            .flatten_to(1)
            .context("failed to flatten Chebyshev weights")?;
        let cheb_term = cheb_basis
            .matmul(&cheb_weights)
            .context("Chebyshev term matmul failed")?;

        let phi = (&base_term + &cheb_term)?;
        phi.broadcast_add(&self.b_phi)
            .context("failed to add phi bias")
    }

    pub fn forward_step(&self, x: &Tensor, h_prev: &Tensor) -> Result<Tensor> {
        let (batch_x, in_dim) = x
            .dims2()
            .context("expected x to be rank-2 [batch, in_dim] in forward_step")?;
        if in_dim != self.cfg.in_dim {
            bail!(
                "x in_dim mismatch: expected {}, got {}",
                self.cfg.in_dim,
                in_dim
            );
        }

        let (batch_h, hidden_dim) = h_prev
            .dims2()
            .context("expected h_prev to be rank-2 [batch, hidden_dim] in forward_step")?;
        if batch_h != batch_x {
            bail!(
                "batch size mismatch: x has {}, h_prev has {}",
                batch_x,
                batch_h
            );
        }
        if hidden_dim != self.cfg.hidden_dim {
            bail!(
                "h_prev hidden_dim mismatch: expected {}, got {}",
                self.cfg.hidden_dim,
                hidden_dim
            );
        }

        let phi = self.phi(x)?;
        let tau = self.tau()?.reshape((1, self.cfg.hidden_dim))?;

        // Euler step for dh/dt = -h/tau + phi(x).
        let leak = h_prev.broadcast_div(&tau)?;
        let dh = (&phi - &leak)?;
        let h_new = (h_prev + ((&dh * self.cfg.dt)?))?;
        Ok(h_new)
    }

    pub fn predict(&self, h: &Tensor) -> Result<Tensor> {
        let (_batch, hidden_dim) = h
            .dims2()
            .context("expected h to be rank-2 [batch, hidden_dim] in predict")?;
        if hidden_dim != self.cfg.hidden_dim {
            bail!(
                "predict hidden_dim mismatch: expected {}, got {}",
                self.cfg.hidden_dim,
                hidden_dim
            );
        }

        let y = h
            .matmul(&self.w_out)
            .context("output projection matmul failed")?;
        y.broadcast_add(&self.b_out)
            .context("failed to add output bias")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn make_layer(cfg: LiquidKanConfig) -> Result<LiquidKanLayer> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        LiquidKanLayer::new(vb, cfg)
    }

    #[test]
    fn chebyshev_basis_shape_ok() -> Result<()> {
        let cfg = LiquidKanConfig {
            in_dim: 1,
            hidden_dim: 4,
            out_dim: 1,
            cheb_order: 3,
            dt: 0.05,
            tau_min: 1e-3,
            x_scale: 2.0,
        };
        let layer = make_layer(cfg.clone())?;
        let x = Tensor::zeros((5, cfg.in_dim), DType::F32, &Device::Cpu)?;
        let basis = layer.chebyshev_basis(&x)?;
        assert_eq!(basis.dims3()?, (5, cfg.in_dim, cfg.cheb_order + 1));
        Ok(())
    }

    #[test]
    fn tau_positive() -> Result<()> {
        let cfg = LiquidKanConfig::default();
        let layer = make_layer(cfg)?;
        let tau = layer.tau()?;
        let tau_min = tau.min_all()?.to_scalar::<f32>()?;
        assert!(tau_min > 0.0);
        Ok(())
    }

    #[test]
    fn forward_step_shape_ok() -> Result<()> {
        let cfg = LiquidKanConfig::default();
        let layer = make_layer(cfg.clone())?;
        let x = Tensor::zeros((7, cfg.in_dim), DType::F32, &Device::Cpu)?;
        let h = layer.zero_state(7)?;
        let h_new = layer.forward_step(&x, &h)?;
        assert_eq!(h_new.dims2()?, (7, cfg.hidden_dim));
        Ok(())
    }

    #[test]
    fn predict_shape_ok() -> Result<()> {
        let cfg = LiquidKanConfig::default();
        let layer = make_layer(cfg.clone())?;
        let h = layer.zero_state(9)?;
        let y = layer.predict(&h)?;
        assert_eq!(y.dims2()?, (9, cfg.out_dim));
        Ok(())
    }

    #[test]
    fn invalid_shape_returns_error() -> Result<()> {
        let cfg = LiquidKanConfig::default();
        let layer = make_layer(cfg.clone())?;
        let x_bad = Tensor::zeros((4, cfg.in_dim + 1), DType::F32, &Device::Cpu)?;
        let h = layer.zero_state(4)?;
        assert!(layer.forward_step(&x_bad, &h).is_err());
        Ok(())
    }
}
