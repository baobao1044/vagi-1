//! Affective Latent Space — Emotion vectors for response modulation.
//!
//! Maps emotional state into a continuous 4D space:
//! - **Valence**: negative ↔ positive
//! - **Arousal**: calm ↔ excited
//! - **Dominance**: submissive ↔ dominant
//! - **Trust**: cautious ↔ trusting
//!
//! Provides:
//! - Keyword-based emotion detection from user text
//! - Smooth interpolation toward target emotion (max Δ = 0.3/turn)
//! - Response modulation metadata (warmth, formality, encouragement)

use serde::{Deserialize, Serialize};

// ─── Emotion Vector ─────────────────────────────────────────────────────────

/// A point in the 4D emotional space. Each dimension in [-1.0, 1.0].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionVector {
    pub valence: f32,   // -1 = sad/negative, +1 = happy/positive
    pub arousal: f32,   // -1 = calm/tired, +1 = excited/energized
    pub dominance: f32, // -1 = helpless, +1 = empowered
    pub trust: f32,     // -1 = suspicious, +1 = trusting
}

impl Default for EmotionVector {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            trust: 0.0,
        }
    }
}

impl EmotionVector {
    pub fn new(valence: f32, arousal: f32, dominance: f32, trust: f32) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
            trust: trust.clamp(-1.0, 1.0),
        }
    }

    /// Euclidean distance to another emotion vector.
    pub fn distance(&self, other: &Self) -> f32 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        let dt = self.trust - other.trust;
        (dv * dv + da * da + dd * dd + dt * dt).sqrt()
    }

    /// Move toward `target` by at most `max_step` (Euclidean distance).
    /// Returns the new position — never jumps more than max_step.
    pub fn step_toward(&self, target: &Self, max_step: f32) -> Self {
        let dist = self.distance(target);
        if dist <= max_step || dist < 1e-6 {
            return target.clone();
        }
        let ratio = max_step / dist;
        Self::new(
            self.valence + (target.valence - self.valence) * ratio,
            self.arousal + (target.arousal - self.arousal) * ratio,
            self.dominance + (target.dominance - self.dominance) * ratio,
            self.trust + (target.trust - self.trust) * ratio,
        )
    }

    /// Name the dominant emotional category.
    pub fn label(&self) -> &'static str {
        if self.valence > 0.5 && self.arousal > 0.3 {
            "joyful"
        } else if self.valence > 0.3 && self.trust > 0.3 {
            "content"
        } else if self.valence < -0.5 && self.arousal > 0.3 {
            "angry"
        } else if self.valence < -0.5 && self.arousal < -0.3 {
            "sad"
        } else if self.valence < -0.3 && self.dominance < -0.3 {
            "helpless"
        } else if self.arousal > 0.5 && self.dominance > 0.3 {
            "assertive"
        } else if self.arousal < -0.5 {
            "tired"
        } else if self.trust < -0.5 {
            "suspicious"
        } else {
            "neutral"
        }
    }
}

// ─── Response Modulation ─────────────────────────────────────────────────────

/// Metadata controlling response tone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseModulation {
    /// 0.0 = cold/formal, 1.0 = warm/empathetic
    pub warmth: f32,
    /// 0.0 = casual, 1.0 = highly formal
    pub formality: f32,
    /// 0.0 = no encouragement, 1.0 = strongly encouraging
    pub encouragement: f32,
    /// 0.0 = no humor, 1.0 = humorous
    pub humor: f32,
    /// Suggested response prefix or framing.
    pub suggested_framing: String,
}

// ─── Affect Engine ───────────────────────────────────────────────────────────

/// Stateless affect processor.
pub struct AffectEngine;

impl AffectEngine {
    pub fn new() -> Self {
        Self
    }

    /// Detect user emotion from text using keyword/pattern matching.
    pub fn detect_emotion(&self, text: &str) -> EmotionVector {
        let lower = text.to_lowercase();
        let mut valence = 0.0_f32;
        let mut arousal = 0.0_f32;
        let mut dominance = 0.0_f32;
        let mut trust = 0.0_f32;
        let mut signals = 0_u32;

        // Positive valence signals
        for pat in &[
            "cảm ơn",
            "thank",
            "tuyệt",
            "great",
            "awesome",
            "love",
            "perfect",
            "wonderful",
            "excellent",
            "nice",
            "good",
            "hay",
            "giỏi",
            "tốt",
            "vui",
            "happy",
            "haha",
            "😊",
            "😀",
            "👍",
            "cool",
            "amazing",
            "impressive",
            "brilliant",
        ] {
            if lower.contains(pat) {
                valence += 0.25;
                signals += 1;
            }
        }

        // Negative valence signals
        for pat in &[
            "bug",
            "lỗi",
            "error",
            "fail",
            "broken",
            "suck",
            "bad",
            "hate",
            "terrible",
            "buồn",
            "sad",
            "angry",
            "frustrated",
            "annoying",
            "chán",
            "mệt",
            "tired",
            "ugh",
            "😡",
            "😢",
            "damn",
            "shit",
            "wtf",
            "fix this",
            "không được",
        ] {
            if lower.contains(pat) {
                valence -= 0.25;
                signals += 1;
            }
        }

        // High arousal signals
        for pat in &[
            "urgent",
            "asap",
            "gấp",
            "nhanh",
            "hurry",
            "now",
            "excited",
            "!!!",
            "!!",
            "emergency",
            "critical",
        ] {
            if lower.contains(pat) {
                arousal += 0.30;
                signals += 1;
            }
        }

        // Low arousal signals
        for pat in &[
            "chill",
            "relax",
            "slow",
            "no rush",
            "từ từ",
            "thong thả",
            "nhẹ nhàng",
            "bored",
            "chán",
            "meh",
            "whatever",
        ] {
            if lower.contains(pat) {
                arousal -= 0.25;
                signals += 1;
            }
        }

        // Dominance signals (user feeling empowered)
        for pat in &[
            "i want",
            "tôi muốn",
            "do this",
            "make it",
            "change",
            "must",
            "need",
            "require",
            "command",
            "order",
        ] {
            if lower.contains(pat) {
                dominance += 0.20;
                signals += 1;
            }
        }

        // Helplessness signals
        for pat in &[
            "help",
            "giúp",
            "stuck",
            "don't know",
            "confused",
            "không biết",
            "lost",
            "how do i",
            "can you",
        ] {
            if lower.contains(pat) {
                dominance -= 0.20;
                trust += 0.15; // Asking for help implies some trust.
                signals += 1;
            }
        }

        // Trust signals
        for pat in &["trust", "tin", "rely", "depend", "always", "favorite"] {
            if lower.contains(pat) {
                trust += 0.25;
                signals += 1;
            }
        }

        // Distrust signals
        for pat in &[
            "wrong",
            "sai",
            "incorrect",
            "lie",
            "doubt",
            "sure?",
            "really?",
            "are you sure",
        ] {
            if lower.contains(pat) {
                trust -= 0.20;
                signals += 1;
            }
        }

        // If no patterns matched, return neutral.
        if signals == 0 {
            return EmotionVector::default();
        }

        EmotionVector::new(
            valence.clamp(-1.0, 1.0),
            arousal.clamp(-1.0, 1.0),
            dominance.clamp(-1.0, 1.0),
            trust.clamp(-1.0, 1.0),
        )
    }

    /// Compute the target emotion we want to help the user reach.
    /// Strategy: gently move toward positive-calm-empowered-trusting.
    pub fn compute_target(&self, current: &EmotionVector) -> EmotionVector {
        // Default target: content, calm, empowered, trusting.
        let default_target = EmotionVector::new(0.5, 0.0, 0.3, 0.5);

        // If user is already positive, maintain. Don't force calmness on excited-happy.
        if current.valence > 0.3 && current.arousal > 0.3 {
            // They're happy and excited — stay with them.
            return EmotionVector::new(0.6, 0.2, 0.3, 0.5);
        }

        // If user is sad/frustrated, target gentle uplift.
        if current.valence < -0.3 {
            return EmotionVector::new(0.2, -0.1, 0.2, 0.4);
        }

        default_target
    }

    /// Plan the response tone by stepping from current toward target.
    /// Max step = 0.3 per turn to avoid jarring emotional jumps.
    pub fn plan_response_tone(
        &self,
        current_user: &EmotionVector,
    ) -> (EmotionVector, ResponseModulation) {
        let target = self.compute_target(current_user);
        let response_emotion = current_user.step_toward(&target, 0.3);

        let modulation = self.derive_modulation(current_user, &response_emotion);
        (response_emotion, modulation)
    }

    /// Derive response modulation from emotional context.
    fn derive_modulation(
        &self,
        user_emotion: &EmotionVector,
        response_emotion: &EmotionVector,
    ) -> ResponseModulation {
        // Warmth increases when user is sad/frustrated.
        let warmth = (0.5 - user_emotion.valence * 0.3).clamp(0.0, 1.0);

        // Formality decreases when trust is high.
        let formality = (0.5 - response_emotion.trust * 0.3).clamp(0.0, 1.0);

        // Encouragement when user feels helpless.
        let encouragement = (0.5 - user_emotion.dominance * 0.4).clamp(0.0, 1.0);

        // Humor only when user is positive and calm.
        let humor = if user_emotion.valence > 0.2 && user_emotion.arousal < 0.3 {
            0.3
        } else {
            0.0
        };

        // Suggested framing based on emotion.
        let framing = match user_emotion.label() {
            "sad" | "helpless" => "Tôi hiểu, mình sẽ cùng giải quyết nhé.",
            "angry" | "frustrated" => "Tôi hiểu sự bực bội. Để tôi xem xét kỹ hơn.",
            "tired" => "Không vội, mình từ từ nhé.",
            "joyful" | "content" => "Tuyệt vời! Tiếp tục nào!",
            "suspicious" => "Tôi sẽ giải thích cặn kẽ từng bước.",
            _ => "",
        };

        ResponseModulation {
            warmth,
            formality,
            encouragement,
            humor,
            suggested_framing: framing.to_string(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_text_gives_neutral_emotion() {
        let engine = AffectEngine::new();
        let em = engine.detect_emotion("please process this data");
        assert!(em.valence.abs() < 0.5);
    }

    #[test]
    fn positive_text_gives_positive_valence() {
        let engine = AffectEngine::new();
        let em = engine.detect_emotion("thank you so much, this is awesome!");
        assert!(
            em.valence > 0.3,
            "valence should be positive: {}",
            em.valence
        );
    }

    #[test]
    fn negative_text_gives_negative_valence() {
        let engine = AffectEngine::new();
        let em = engine.detect_emotion("this is broken, there are so many bugs, I hate it");
        assert!(
            em.valence < -0.3,
            "valence should be negative: {}",
            em.valence
        );
    }

    #[test]
    fn step_toward_respects_max_step() {
        let current = EmotionVector::new(-0.8, 0.5, -0.5, -0.3);
        let target = EmotionVector::new(0.5, 0.0, 0.3, 0.5);
        let stepped = current.step_toward(&target, 0.3);
        let dist = current.distance(&stepped);
        assert!(dist <= 0.31, "step should not exceed max_step: dist={dist}");
    }

    #[test]
    fn step_toward_reaches_target_when_close() {
        let current = EmotionVector::new(0.49, 0.01, 0.29, 0.49);
        let target = EmotionVector::new(0.50, 0.00, 0.30, 0.50);
        let stepped = current.step_toward(&target, 0.3);
        assert!((stepped.valence - 0.5).abs() < 0.01);
    }

    #[test]
    fn modulation_warmth_increases_for_sad_user() {
        let engine = AffectEngine::new();
        let sad = EmotionVector::new(-0.7, -0.3, -0.3, 0.0);
        let happy = EmotionVector::new(0.7, 0.3, 0.3, 0.5);
        let (_, mod_sad) = engine.plan_response_tone(&sad);
        let (_, mod_happy) = engine.plan_response_tone(&happy);
        assert!(
            mod_sad.warmth > mod_happy.warmth,
            "sad user should get more warmth: sad={} happy={}",
            mod_sad.warmth,
            mod_happy.warmth
        );
    }

    #[test]
    fn vietnamese_keywords_detected() {
        let engine = AffectEngine::new();
        let em = engine.detect_emotion("cảm ơn bạn, tuyệt vời quá!");
        assert!(em.valence > 0.3);
    }
}
