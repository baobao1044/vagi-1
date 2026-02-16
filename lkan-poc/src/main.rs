mod model;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use model::lkan::{LiquidKanConfig, LiquidKanLayer};
use rand::seq::SliceRandom;

fn build_dataset(n_samples: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut xs = Vec::with_capacity(n_samples);
    let mut ys = Vec::with_capacity(n_samples);
    let denom = (n_samples.saturating_sub(1)) as f32;

    for idx in 0..n_samples {
        let ratio = if denom > 0.0 { idx as f32 / denom } else { 0.0 };
        let x = -2.0_f32 + 4.0_f32 * ratio;
        xs.push(x);
        ys.push(x.sin() * x);
    }

    let x = Tensor::from_slice(&xs, (n_samples, 1), device).context("failed to build x tensor")?;
    let y = Tensor::from_slice(&ys, (n_samples, 1), device).context("failed to build y tensor")?;
    Ok((x, y))
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let (x_all, y_all) = build_dataset(512, &device)?;

    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let cfg = LiquidKanConfig {
        in_dim: 1,
        hidden_dim: 8,
        out_dim: 1,
        cheb_order: 5,
        dt: 0.05,
        tau_min: 1e-3,
        x_scale: 2.0,
    };
    let layer = LiquidKanLayer::new(vb.pp("model"), cfg.clone())?;

    let mut optimizer = AdamW::new(
        var_map.all_vars(),
        ParamsAdamW {
            lr: 1e-2,
            weight_decay: 1e-4,
            ..Default::default()
        },
    )?;

    let epochs = 3_000;
    let batch_size = 64;
    let n_samples = x_all.dim(0)?;
    let mut shuffled_indices: Vec<u32> = (0..n_samples as u32).collect();
    let mut rng = rand::rng();

    for epoch in 1..=epochs {
        shuffled_indices.shuffle(&mut rng);
        let mut loss_sum = 0.0_f64;
        let mut steps = 0usize;

        for start in (0..n_samples).step_by(batch_size) {
            let end = (start + batch_size).min(n_samples);
            let batch_ids = &shuffled_indices[start..end];
            let idx = Tensor::from_slice(batch_ids, (batch_ids.len(),), &device)?;

            let xb = x_all.index_select(&idx, 0)?;
            let yb = y_all.index_select(&idx, 0)?;

            let h0 = layer.zero_state(batch_ids.len())?;
            let h1 = layer.forward_step(&xb, &h0)?;
            let pred = layer.predict(&h1)?;

            let diff = (&pred - &yb)?;
            let loss = diff.sqr()?.mean_all()?;
            let loss_value = loss.to_scalar::<f32>()? as f64;
            optimizer.backward_step(&loss)?;

            loss_sum += loss_value;
            steps += 1;
        }

        if epoch % 100 == 0 {
            println!(
                "epoch {:4} | train_mse {:.6}",
                epoch,
                loss_sum / steps as f64
            );
        }
    }

    let h_eval = layer.zero_state(n_samples)?;
    let h_eval = layer.forward_step(&x_all, &h_eval)?;
    let pred_all = layer.predict(&h_eval)?;
    let final_mse = (&pred_all - &y_all)?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?;
    println!("final mse: {:.6}", final_mse);

    let n_show = 5usize.min(n_samples);
    let x_show = x_all.narrow(0, 0, n_show)?.reshape((n_show,))?;
    let y_show = y_all.narrow(0, 0, n_show)?.reshape((n_show,))?;
    let p_show = pred_all.narrow(0, 0, n_show)?.reshape((n_show,))?;

    let x_vals = x_show.to_vec1::<f32>()?;
    let y_vals = y_show.to_vec1::<f32>()?;
    let p_vals = p_show.to_vec1::<f32>()?;

    println!("sample predictions:");
    for i in 0..n_show {
        println!(
            "x={:+.4} | y_true={:+.4} | y_pred={:+.4}",
            x_vals[i], y_vals[i], p_vals[i]
        );
    }

    Ok(())
}
