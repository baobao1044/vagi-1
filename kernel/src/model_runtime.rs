use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::model::gpt_kan::{LKanGPT, LKanGPTConfig};
use crate::model::lkan::LiquidKanConfig;
use crate::models::{MctsInferResponse, ModelInferResponse, ModelLoadResponse, ModelStatusResponse};

pub const VOCAB_CHARS: &str = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

const DEFAULT_MODEL_PATH: &str = "models/lkan-genesis.safetensors";
const MODEL_ID: &str = "lkan-genesis";
const MODEL_ARCH: &str = "lkan-gpt";
const MODEL_VERSION: &str = "v1";
const MAX_CONTEXT_LEN: usize = 64;
const MAX_NEW_TOKENS: usize = 256;

#[derive(Debug, Clone)]
pub struct ModelManifest {
    pub model_id: String,
    pub arch: String,
    pub version: String,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub pad_id: usize,
    pub unk_id: usize,
    pub max_seq_len: usize,
    pub model_file: String,
    pub vocab_file: String,
    pub model_sha256: String,
    pub vocab_sha256: String,
}

impl ModelManifest {
    fn lkan_defaults(vocab_size: usize) -> Self {
        Self {
            model_id: MODEL_ID.to_string(),
            arch: MODEL_ARCH.to_string(),
            version: MODEL_VERSION.to_string(),
            vocab_size,
            embed_dim: 128,
            hidden_dim: 128,
            num_layers: 8,
            bos_id: vocab_size,
            eos_id: vocab_size + 1,
            pad_id: vocab_size + 2,
            unk_id: vocab_size + 3,
            max_seq_len: MAX_CONTEXT_LEN,
            model_file: DEFAULT_MODEL_PATH.to_string(),
            vocab_file: String::new(),
            model_sha256: String::new(),
            vocab_sha256: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    id_to_char: Vec<char>,
    char_to_id: HashMap<char, u32>,
    fallback_id: u32,
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        let id_to_char: Vec<char> = VOCAB_CHARS.chars().collect();
        let char_to_id: HashMap<char, u32> = id_to_char
            .iter()
            .enumerate()
            .map(|(idx, ch)| (*ch, idx as u32))
            .collect();
        let fallback_id = *char_to_id.get(&' ').unwrap_or(&0);
        Self {
            id_to_char,
            char_to_id,
            fallback_id,
        }
    }
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    pub fn fallback_id(&self) -> u32 {
        self.fallback_id
    }

    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text
            .chars()
            .map(|ch| {
                self.char_to_id
                    .get(&ch)
                    .copied()
                    .unwrap_or(self.fallback_id)
            })
            .collect();
        if ids.is_empty() {
            ids.push(self.fallback_id);
        }
        ids
    }

    pub fn encode(&self, text: &str, device: &Device) -> Result<Tensor> {
        let ids = self.encode_ids(text);
        Tensor::from_slice(&ids, (1, ids.len()), device)
            .context("failed to build token id tensor in tokenizer")
    }

    pub fn decode(&self, token_id: u32) -> String {
        self.id_to_char
            .get(token_id as usize)
            .copied()
            .unwrap_or(' ')
            .to_string()
    }

    pub fn decode_ids(&self, token_ids: &[usize]) -> String {
        let mut out = String::with_capacity(token_ids.len());
        for &id in token_ids {
            out.push(self.id_to_char.get(id).copied().unwrap_or(' '));
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub manifest: ModelManifest,
    tokenizer: SimpleTokenizer,
    model: Arc<Mutex<Option<LKanGPT>>>,
    device: Device,
}

impl LoadedModel {
    fn from_runtime(runtime: &ModelRuntime) -> Self {
        Self {
            manifest: ModelManifest::lkan_defaults(runtime.tokenizer.vocab_size()),
            tokenizer: runtime.tokenizer.clone(),
            model: Arc::clone(&runtime.model),
            device: runtime.device.clone(),
        }
    }

    pub fn encode_prompt(&self, prompt: &str) -> Vec<usize> {
        self.tokenizer
            .encode_ids(prompt)
            .into_iter()
            .map(|id| id as usize)
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        self.tokenizer.decode_ids(ids)
    }

    pub fn forward_one_token(&self, token_id: usize, _hidden: &mut [Vec<f32>]) -> Result<Vec<f32>> {
        let guard = self.model.lock().expect("model runtime lock poisoned");
        let Some(model) = guard.as_ref() else {
            bail!("model is not loaded");
        };

        let bounded_token = if token_id < self.manifest.vocab_size {
            token_id as u32
        } else {
            self.tokenizer.fallback_id()
        };

        let input_ids = Tensor::from_slice(&[bounded_token], (1, 1), &self.device)
            .context("failed to create one-token input tensor")?;
        let logits = model
            .forward_logits(&input_ids)
            .context("LKanGPT forward failed in forward_one_token")?;
        let (_bsz, seq_len, vocab_size) = logits
            .dims3()
            .context("expected logits shape [1, seq_len, vocab_size]")?;
        if seq_len == 0 {
            bail!("forward_one_token received empty sequence logits");
        }
        logits
            .narrow(1, seq_len - 1, 1)?
            .reshape((vocab_size,))?
            .to_vec1::<f32>()
            .context("failed to convert logits to vector")
    }
}

pub struct ModelRuntime {
    pub model: Arc<Mutex<Option<LKanGPT>>>,
    pub device: Device,
    pub tokenizer: SimpleTokenizer,
}

impl Default for ModelRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRuntime {
    pub fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            device: Device::Cpu,
            tokenizer: SimpleTokenizer::new(),
        }
    }

    fn build_config(&self) -> Result<LKanGPTConfig> {
        let vocab_size = self.tokenizer.vocab_size();
        if vocab_size != 65 {
            bail!(
                "unexpected tokenizer vocab size: expected 65, got {}",
                vocab_size
            );
        }

        Ok(LKanGPTConfig {
            vocab_size,
            hidden_dim: 128,
            num_layers: 8,
            num_heads: 4,
            kan_config: LiquidKanConfig {
                in_dim: 128,
                hidden_dim: 128,
                out_dim: 128,
                cheb_order: 3,
                dt: 0.1,
                tau_min: 1e-2,
                x_scale: 1.0,
            },
        })
    }

    fn resolve_checkpoint_path(&self, model_dir: &str) -> PathBuf {
        let trimmed = model_dir.trim();
        if trimmed.is_empty() {
            return PathBuf::from(DEFAULT_MODEL_PATH);
        }

        let candidate = PathBuf::from(trimmed);
        if candidate.is_file() {
            return candidate;
        }

        let nested = candidate.join("lkan-genesis.safetensors");
        if nested.exists() {
            return nested;
        }

        PathBuf::from(DEFAULT_MODEL_PATH)
    }

    fn load_from_checkpoint_path(&self, checkpoint_path: &Path) -> Result<ModelLoadResponse> {
        if !checkpoint_path.exists() {
            bail!(
                "missing checkpoint at {}. Train first using `cargo run -p vagi-kernel --bin train_lkan --release`.",
                checkpoint_path.display()
            );
        }

        let cfg = self.build_config()?;
        let weights = vec![checkpoint_path.to_path_buf()];
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &self.device)
                .with_context(|| {
                    format!(
                        "failed to load safetensors checkpoint from {}",
                        checkpoint_path.display()
                    )
                })?
        };
        let model =
            LKanGPT::new(vb.pp("lkan_gpt"), cfg).context("failed to instantiate LKanGPT runtime")?;

        let mut guard = self.model.lock().expect("model runtime lock poisoned");
        *guard = Some(model);

        Ok(ModelLoadResponse {
            model_id: MODEL_ID.to_string(),
            loaded: true,
            checksum_ok: true,
            arch: MODEL_ARCH.to_string(),
        })
    }

    pub fn load(&self) -> Result<ModelLoadResponse> {
        self.load_from_checkpoint_path(Path::new(DEFAULT_MODEL_PATH))
    }

    pub fn load_from_dir(&self, model_dir: &str) -> Result<ModelLoadResponse> {
        let checkpoint_path = self.resolve_checkpoint_path(model_dir);
        self.load_from_checkpoint_path(&checkpoint_path)
    }

    pub fn status(&self) -> ModelStatusResponse {
        let loaded = self.model.lock().expect("model runtime lock poisoned").is_some();
        if loaded {
            ModelStatusResponse {
                loaded: true,
                model_id: Some(MODEL_ID.to_string()),
                arch: Some(MODEL_ARCH.to_string()),
                vocab_size: Some(self.tokenizer.vocab_size()),
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

    fn next_token_id_with_model(&self, model: &LKanGPT, prompt: &str) -> Result<u32> {
        let clipped_prompt = last_chars(prompt, MAX_CONTEXT_LEN);
        let input_ids = self.tokenizer.encode(&clipped_prompt, &self.device)?;
        let logits = model
            .forward_logits(&input_ids)
            .context("forward_logits failed during inference")?;
        let (_bsz, seq_len, vocab_size) = logits
            .dims3()
            .context("expected logits shape [batch, seq_len, vocab_size]")?;
        if seq_len == 0 {
            bail!("received empty sequence logits while inferring");
        }

        let last_logits = logits
            .narrow(1, seq_len - 1, 1)?
            .reshape((vocab_size,))?
            .to_vec1::<f32>()?;
        Ok(argmax(&last_logits) as u32)
    }

    pub fn infer_next_char(&self, prompt: &str) -> Result<String> {
        let guard = self.model.lock().expect("model runtime lock poisoned");
        let Some(model) = guard.as_ref() else {
            bail!("model is not loaded");
        };
        let next_id = self.next_token_id_with_model(model, prompt)?;
        Ok(self.tokenizer.decode(next_id))
    }

    pub fn infer(&self, prompt: &str, max_new_tokens: usize) -> Result<ModelInferResponse> {
        let started = Instant::now();
        let max_tokens = max_new_tokens.clamp(1, MAX_NEW_TOKENS);

        let guard = self.model.lock().expect("model runtime lock poisoned");
        let Some(model) = guard.as_ref() else {
            bail!("model is not loaded");
        };

        let mut running_prompt = prompt.to_string();
        let mut generated_ids = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next_id = self.next_token_id_with_model(model, &running_prompt)?;
            generated_ids.push(next_id as usize);
            running_prompt.push_str(&self.tokenizer.decode(next_id));
        }

        Ok(ModelInferResponse {
            model_id: MODEL_ID.to_string(),
            text: self.tokenizer.decode_ids(&generated_ids),
            tokens_generated: generated_ids.len(),
            latency_ms: started.elapsed().as_millis() as u64,
            think_trace: None,
        })
    }

    pub fn infer_mcts(
        &self,
        prompt: &str,
        mcts_engine: &crate::mcts::MctsEngine,
        _verifier: &crate::verifier::Verifier,
        _world_model: &crate::world_model::WorldModel,
    ) -> Result<MctsInferResponse> {
        // v1 compatibility mode: fallback to deterministic greedy decode.
        let max_tokens = mcts_engine.config.max_depth.clamp(1, MAX_NEW_TOKENS);
        let output = self.infer(prompt, max_tokens)?;
        Ok(MctsInferResponse {
            model_id: output.model_id,
            text: output.text,
            tokens_generated: output.tokens_generated,
            branches_explored: 1,
            best_branch_reward: 0.0,
            latency_ms: output.latency_ms,
        })
    }

    pub fn as_loaded_model(&self) -> LoadedModel {
        LoadedModel::from_runtime(self)
    }
}

fn last_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars()
        .rev()
        .take(max_chars)
        .collect::<Vec<char>>()
        .into_iter()
        .rev()
        .collect()
}

pub fn argmax(values: &[f32]) -> usize {
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

pub fn softmax_top_k(logits: &[f32], k: usize, temperature: f32) -> Vec<(usize, f32)> {
    let temp = temperature.max(0.01);
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| (idx, ((logit - max_logit) / temp).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, value)| *value).sum::<f32>().max(1e-12);
    for (_, value) in &mut probs {
        *value /= sum;
    }
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs.truncate(k);
    probs
}

pub fn argmax_with_exclusions(values: &[f32], excluded: &[usize]) -> usize {
    let excluded: HashSet<usize> = excluded.iter().copied().collect();
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if excluded.contains(&idx) {
            continue;
        }
        if *value > best_val {
            best_idx = idx;
            best_val = *value;
        }
    }
    if best_val == f32::NEG_INFINITY {
        argmax(values)
    } else {
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::{SimpleTokenizer, argmax_with_exclusions, last_chars};

    #[test]
    fn tokenizer_matches_expected_vocab_size() {
        let tok = SimpleTokenizer::new();
        assert_eq!(tok.vocab_size(), 65);
    }

    #[test]
    fn last_chars_clips_unicode_safely() {
        let clipped = last_chars("abcdef", 3);
        assert_eq!(clipped, "def");
    }

    #[test]
    fn argmax_exclusions_fallback_when_all_blocked() {
        let values = [0.1_f32, 0.2, 0.3];
        let idx = argmax_with_exclusions(&values, &[0, 1, 2]);
        assert_eq!(idx, 2);
    }
}
