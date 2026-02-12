use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::RwLock;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::models::{ModelInferResponse, ModelLoadResponse, ModelStatusResponse};

#[derive(Debug, Deserialize, Clone)]
struct ModelManifest {
    model_id: String,
    arch: String,
    version: String,
    vocab_size: usize,
    embed_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
    bos_id: usize,
    eos_id: usize,
    pad_id: usize,
    unk_id: usize,
    max_seq_len: usize,
    model_file: String,
    vocab_file: String,
    model_sha256: String,
    vocab_sha256: String,
}

#[derive(Debug, Deserialize)]
struct VocabFile {
    tokens: Vec<String>,
}

#[derive(Debug, Clone)]
struct GruLayerWeights {
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
    input_size: usize,
}

#[derive(Debug, Clone)]
struct LoadedModel {
    manifest: ModelManifest,
    tokens: Vec<String>,
    token_to_id: HashMap<String, usize>,
    embedding: Vec<f32>,
    gru_layers: Vec<GruLayerWeights>,
    lm_head_weight: Vec<f32>,
    lm_head_bias: Vec<f32>,
}

#[derive(Default)]
pub struct ModelRuntime {
    loaded: RwLock<Option<LoadedModel>>,
}

impl ModelRuntime {
    pub fn new() -> Self {
        Self {
            loaded: RwLock::new(None),
        }
    }

    pub fn load_from_dir(&self, model_dir: &str) -> Result<ModelLoadResponse> {
        let dir = Path::new(model_dir);
        let manifest_path = dir.join("manifest.json");
        let manifest: ModelManifest = serde_json::from_slice(
            &fs::read(&manifest_path).with_context(|| {
                format!("failed to read manifest at {}", manifest_path.display())
            })?,
        )?;

        if manifest.arch != "tiny-gru-lm" {
            bail!("unsupported model arch `{}`", manifest.arch);
        }
        if manifest.vocab_size == 0 || manifest.embed_dim == 0 || manifest.hidden_dim == 0 {
            bail!("invalid manifest dimensions");
        }
        if manifest.num_layers == 0 {
            bail!("manifest.num_layers must be > 0");
        }
        let _ = &manifest.version;
        let _ = manifest.max_seq_len;

        let model_path = dir.join(&manifest.model_file);
        let vocab_path = dir.join(&manifest.vocab_file);
        let model_checksum = sha256_file(&model_path)?;
        let vocab_checksum = sha256_file(&vocab_path)?;
        if model_checksum != manifest.model_sha256 {
            bail!("model checksum mismatch");
        }
        if vocab_checksum != manifest.vocab_sha256 {
            bail!("vocab checksum mismatch");
        }

        let vocab: VocabFile = serde_json::from_slice(
            &fs::read(&vocab_path)
                .with_context(|| format!("failed to read vocab at {}", vocab_path.display()))?,
        )?;
        if vocab.tokens.len() != manifest.vocab_size {
            bail!(
                "vocab size mismatch: manifest={} file={}",
                manifest.vocab_size,
                vocab.tokens.len()
            );
        }

        let safetensor_bytes = fs::read(&model_path)
            .with_context(|| format!("failed to read model at {}", model_path.display()))?;
        let tensors = SafeTensors::deserialize(&safetensor_bytes)?;

        let embedding = load_matrix(
            &tensors,
            "embedding.weight",
            manifest.vocab_size,
            manifest.embed_dim,
        )?;

        let mut gru_layers = Vec::with_capacity(manifest.num_layers);
        for layer_idx in 0..manifest.num_layers {
            let input_size = if layer_idx == 0 {
                manifest.embed_dim
            } else {
                manifest.hidden_dim
            };
            let rows = manifest.hidden_dim * 3;
            let weight_ih = load_matrix(
                &tensors,
                &format!("gru.weight_ih_l{layer_idx}"),
                rows,
                input_size,
            )?;
            let weight_hh = load_matrix(
                &tensors,
                &format!("gru.weight_hh_l{layer_idx}"),
                rows,
                manifest.hidden_dim,
            )?;
            let bias_ih = load_vector(&tensors, &format!("gru.bias_ih_l{layer_idx}"), rows)?;
            let bias_hh = load_vector(&tensors, &format!("gru.bias_hh_l{layer_idx}"), rows)?;
            gru_layers.push(GruLayerWeights {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                input_size,
            });
        }

        let lm_head_weight = load_matrix(
            &tensors,
            "lm_head.weight",
            manifest.vocab_size,
            manifest.hidden_dim,
        )?;
        let lm_head_bias = load_vector(&tensors, "lm_head.bias", manifest.vocab_size)?;

        let token_to_id = vocab
            .tokens
            .iter()
            .enumerate()
            .map(|(idx, token)| (token.clone(), idx))
            .collect();

        let loaded = LoadedModel {
            manifest: manifest.clone(),
            tokens: vocab.tokens,
            token_to_id,
            embedding,
            gru_layers,
            lm_head_weight,
            lm_head_bias,
        };

        let mut guard = self.loaded.write().expect("model runtime lock poisoned");
        *guard = Some(loaded);
        Ok(ModelLoadResponse {
            model_id: manifest.model_id,
            loaded: true,
            checksum_ok: true,
            arch: manifest.arch,
        })
    }

    pub fn status(&self) -> ModelStatusResponse {
        let guard = self.loaded.read().expect("model runtime lock poisoned");
        if let Some(model) = guard.as_ref() {
            ModelStatusResponse {
                loaded: true,
                model_id: Some(model.manifest.model_id.clone()),
                arch: Some(model.manifest.arch.clone()),
                vocab_size: Some(model.manifest.vocab_size),
            }
        } else {
            ModelStatusResponse {
                loaded: false,
                model_id: None,
                arch: None,
                vocab_size: None,
            }
        }
    }

    pub fn infer(&self, prompt: &str, max_new_tokens: usize) -> Result<ModelInferResponse> {
        let guard = self.loaded.read().expect("model runtime lock poisoned");
        let Some(model) = guard.as_ref() else {
            bail!("model is not loaded");
        };
        let started = Instant::now();
        let max_tokens = max_new_tokens.clamp(1, 256);

        let mut hidden = vec![vec![0.0_f32; model.manifest.hidden_dim]; model.manifest.num_layers];
        let input_ids = model.encode_prompt(prompt);
        let mut logits = vec![0.0_f32; model.manifest.vocab_size];
        for token_id in input_ids {
            logits = model.forward_one_token(token_id, &mut hidden)?;
        }

        let mut generated_ids: Vec<usize> = Vec::new();
        let mut next_id = argmax(&logits);
        for _ in 0..max_tokens {
            if next_id == model.manifest.eos_id {
                break;
            }
            generated_ids.push(next_id);
            logits = model.forward_one_token(next_id, &mut hidden)?;
            next_id = argmax(&logits);
        }

        let text = model.decode(&generated_ids);
        Ok(ModelInferResponse {
            model_id: model.manifest.model_id.clone(),
            text,
            tokens_generated: generated_ids.len(),
            latency_ms: started.elapsed().as_millis() as u64,
        })
    }
}

impl LoadedModel {
    fn encode_prompt(&self, prompt: &str) -> Vec<usize> {
        let mut ids = Vec::with_capacity(prompt.chars().count() + 1);
        ids.push(self.manifest.bos_id);
        for ch in prompt.chars() {
            let token = ch.to_string();
            let id = self
                .token_to_id
                .get(&token)
                .copied()
                .unwrap_or(self.manifest.unk_id);
            ids.push(id);
        }
        ids
    }

    fn decode(&self, ids: &[usize]) -> String {
        let special_ids: HashSet<usize> = [
            self.manifest.pad_id,
            self.manifest.bos_id,
            self.manifest.eos_id,
            self.manifest.unk_id,
        ]
        .into_iter()
        .collect();
        let mut out = String::new();
        for id in ids {
            if special_ids.contains(id) {
                continue;
            }
            if let Some(token) = self.tokens.get(*id) {
                out.push_str(token);
            }
        }
        out
    }

    fn forward_one_token(&self, token_id: usize, hidden: &mut [Vec<f32>]) -> Result<Vec<f32>> {
        if token_id >= self.manifest.vocab_size {
            bail!("token id out of range");
        }
        if hidden.len() != self.manifest.num_layers {
            bail!("hidden state layer mismatch");
        }
        let mut x = self.embedding_row(token_id);
        for (layer_idx, layer) in self.gru_layers.iter().enumerate() {
            let h_prev = hidden
                .get(layer_idx)
                .context("missing hidden layer while inferring")?;
            let h_next = gru_step(layer, &x, h_prev, self.manifest.hidden_dim);
            hidden[layer_idx] = h_next.clone();
            x = h_next;
        }
        Ok(self.lm_head(&x))
    }

    fn embedding_row(&self, token_id: usize) -> Vec<f32> {
        let start = token_id * self.manifest.embed_dim;
        let end = start + self.manifest.embed_dim;
        self.embedding[start..end].to_vec()
    }

    fn lm_head(&self, x: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0_f32; self.manifest.vocab_size];
        for (row, out) in logits.iter_mut().enumerate() {
            let base = row * self.manifest.hidden_dim;
            let mut sum = self.lm_head_bias[row];
            for (col, value) in x.iter().enumerate().take(self.manifest.hidden_dim) {
                sum += self.lm_head_weight[base + col] * *value;
            }
            *out = sum;
        }
        logits
    }
}

fn gru_step(layer: &GruLayerWeights, x: &[f32], h_prev: &[f32], hidden_dim: usize) -> Vec<f32> {
    let mut h_next = vec![0.0_f32; hidden_dim];
    for i in 0..hidden_dim {
        let r = sigmoid(
            linear_row(&layer.weight_ih, i, layer.input_size, x)
                + layer.bias_ih[i]
                + linear_row(&layer.weight_hh, i, hidden_dim, h_prev)
                + layer.bias_hh[i],
        );
        let z_idx = i + hidden_dim;
        let z = sigmoid(
            linear_row(&layer.weight_ih, z_idx, layer.input_size, x)
                + layer.bias_ih[z_idx]
                + linear_row(&layer.weight_hh, z_idx, hidden_dim, h_prev)
                + layer.bias_hh[z_idx],
        );
        let n_idx = i + hidden_dim * 2;
        let n_pre = linear_row(&layer.weight_ih, n_idx, layer.input_size, x) + layer.bias_ih[n_idx];
        let n_recur =
            linear_row(&layer.weight_hh, n_idx, hidden_dim, h_prev) + layer.bias_hh[n_idx];
        let n = (n_pre + r * n_recur).tanh();
        h_next[i] = (1.0 - z) * n + z * h_prev[i];
    }
    h_next
}

fn linear_row(weight: &[f32], row: usize, cols: usize, x: &[f32]) -> f32 {
    let offset = row * cols;
    let mut sum = 0.0_f32;
    for col in 0..cols {
        sum += weight[offset + col] * x[col];
    }
    sum
}

#[inline]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn load_matrix(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor `{name}`"))?;
    if tensor.dtype() != Dtype::F32 {
        bail!("tensor `{name}` must be f32");
    }
    let shape = tensor.shape();
    if shape != [expected_rows, expected_cols] {
        bail!(
            "tensor `{name}` shape mismatch: expected [{expected_rows}, {expected_cols}] got {:?}",
            shape
        );
    }
    bytes_to_f32(tensor.data())
}

fn load_vector(tensors: &SafeTensors<'_>, name: &str, expected_len: usize) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor `{name}`"))?;
    if tensor.dtype() != Dtype::F32 {
        bail!("tensor `{name}` must be f32");
    }
    let shape = tensor.shape();
    if shape != [expected_len] {
        bail!(
            "tensor `{name}` shape mismatch: expected [{expected_len}] got {:?}",
            shape
        );
    }
    bytes_to_f32(tensor.data())
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("invalid f32 byte length {}", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if *value > best_val {
            best_idx = idx;
            best_val = *value;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::{GruLayerWeights, gru_step};

    #[test]
    fn gru_step_keeps_hidden_size() {
        let hidden_dim = 4usize;
        let input_size = 3usize;
        let layer = GruLayerWeights {
            weight_ih: vec![0.01; hidden_dim * 3 * input_size],
            weight_hh: vec![0.02; hidden_dim * 3 * hidden_dim],
            bias_ih: vec![0.0; hidden_dim * 3],
            bias_hh: vec![0.0; hidden_dim * 3],
            input_size,
        };
        let x = vec![0.3, -0.2, 0.1];
        let h_prev = vec![0.0; hidden_dim];
        let h_next = gru_step(&layer, &x, &h_prev, hidden_dim);
        assert_eq!(h_next.len(), hidden_dim);
    }
}

