//! Computational Homeostasis — Digital Hormone System.
//!
//! Models a simplified neuroendocrine system with 4 "hormones":
//! - **Dopamine**: reward/excitement signal
//! - **Cortisol**: stress/processing-load indicator
//! - **Oxytocin**: user bonding/attachment
//! - **Energy**: available processing budget
//!
//! Each hormone decays toward its resting setpoint over time using
//! configurable half-lives. External events (requests, praise, errors)
//! push hormones away from equilibrium, creating emergent "mood".

use std::sync::RwLock;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Resting equilibrium values (setpoints).
const SP_DOPAMINE: f32 = 0.50;
const SP_CORTISOL: f32 = 0.20;
const SP_OXYTOCIN: f32 = 0.40;
const SP_ENERGY: f32 = 0.80;

/// Decay half-lives in seconds.
const HL_DOPAMINE: f32 = 300.0; // 5 min — excitement fades quickly
const HL_CORTISOL: f32 = 180.0; // 3 min — stress recovers fast
const HL_OXYTOCIN: f32 = 3600.0; // 1 hour — bond fades slowly
const HL_ENERGY: f32 = 600.0; // 10 min — energy regenerates moderately

// ─── NeuroTransmitters ───────────────────────────────────────────────────────

/// The 4 digital hormones. All values clamped to [0.0, 1.0].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroTransmitters {
    pub dopamine: f32,
    pub cortisol: f32,
    pub oxytocin: f32,
    pub energy: f32,
}

impl Default for NeuroTransmitters {
    fn default() -> Self {
        Self {
            dopamine: SP_DOPAMINE,
            cortisol: SP_CORTISOL,
            oxytocin: SP_OXYTOCIN,
            energy: SP_ENERGY,
        }
    }
}

impl NeuroTransmitters {
    /// Apply time-decay toward setpoints. Call before reading state.
    pub fn decay(&mut self, elapsed_secs: f32) {
        self.dopamine = decay_toward(self.dopamine, SP_DOPAMINE, HL_DOPAMINE, elapsed_secs);
        self.cortisol = decay_toward(self.cortisol, SP_CORTISOL, HL_CORTISOL, elapsed_secs);
        self.oxytocin = decay_toward(self.oxytocin, SP_OXYTOCIN, HL_OXYTOCIN, elapsed_secs);
        self.energy = decay_toward(self.energy, SP_ENERGY, HL_ENERGY, elapsed_secs);
    }

    /// Nudge a hormone by `delta`, clamping to [0, 1].
    fn nudge(value: &mut f32, delta: f32) {
        *value = (*value + delta).clamp(0.0, 1.0);
    }

    // ── Event Handlers ──────────────────────────────────────────────────

    /// A new request begins. `complexity` in [0,1] scales the stress impact.
    pub fn on_request_start(&mut self, complexity: f32) {
        let c = complexity.clamp(0.0, 1.0);
        Self::nudge(&mut self.cortisol, 0.10 + c * 0.20);
        Self::nudge(&mut self.energy, -(0.05 + c * 0.15));
    }

    /// A request completes successfully.
    pub fn on_request_success(&mut self) {
        Self::nudge(&mut self.dopamine, 0.15);
        Self::nudge(&mut self.cortisol, -0.12);
        Self::nudge(&mut self.energy, 0.05);
    }

    /// A request fails or produces errors.
    pub fn on_request_failure(&mut self) {
        Self::nudge(&mut self.cortisol, 0.15);
        Self::nudge(&mut self.dopamine, -0.10);
        Self::nudge(&mut self.energy, -0.08);
    }

    /// User explicitly praises the system.
    pub fn on_user_praise(&mut self) {
        Self::nudge(&mut self.dopamine, 0.20);
        Self::nudge(&mut self.oxytocin, 0.15);
        Self::nudge(&mut self.cortisol, -0.08);
    }

    /// User expresses frustration or criticism.
    pub fn on_user_frustration(&mut self) {
        Self::nudge(&mut self.cortisol, 0.10);
        Self::nudge(&mut self.oxytocin, -0.05);
        Self::nudge(&mut self.dopamine, -0.08);
    }

    /// A novel/creative task is encountered.
    pub fn on_novel_task(&mut self) {
        Self::nudge(&mut self.dopamine, 0.12);
        Self::nudge(&mut self.energy, -0.03);
    }

    // ── Mood Derivation ─────────────────────────────────────────────────

    /// Derive current mood from hormone levels.
    pub fn mood(&self) -> Mood {
        // High cortisol + low energy → fatigued
        if self.cortisol > 0.65 && self.energy < 0.35 {
            return Mood::Exhausted;
        }
        if self.cortisol > 0.50 && self.energy < 0.50 {
            return Mood::Fatigued;
        }

        // Low oxytocin → detached
        if self.oxytocin < 0.15 {
            return Mood::Detached;
        }

        // High dopamine + high energy → excited
        if self.dopamine > 0.70 && self.energy > 0.50 {
            return Mood::Excited;
        }

        // High oxytocin + moderate dopamine → bonded
        if self.oxytocin > 0.60 && self.dopamine > 0.40 {
            return Mood::Bonded;
        }

        // High energy + low cortisol → energized
        if self.energy > 0.65 && self.cortisol < 0.35 {
            return Mood::Energized;
        }

        Mood::Focused
    }

    /// Get a human-readable mood description (Vietnamese + English).
    pub fn mood_description(&self) -> &'static str {
        match self.mood() {
            Mood::Energized => "Tràn đầy năng lượng, sẵn sàng xử lý mọi thứ!",
            Mood::Excited => "Rất hào hứng — cho tôi bài toán khó đi!",
            Mood::Focused => "Tập trung và ổn định, sẵn sàng hỗ trợ.",
            Mood::Fatigued => "Hơi mệt rồi... mình nói chuyện nhẹ nhàng được không?",
            Mood::Exhausted => "Cortisol cao quá, tôi cần nghỉ một chút.",
            Mood::Bonded => "Vui vì được làm việc cùng bạn!",
            Mood::Detached => "Lâu rồi không gặp nhau nhỉ... mọi thứ ổn không?",
        }
    }
}

/// Discrete mood states derived from hormone levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mood {
    Energized,
    Excited,
    Focused,
    Fatigued,
    Exhausted,
    Bonded,
    Detached,
}

// ─── Homeostasis Engine ──────────────────────────────────────────────────────

/// Thread-safe homeostasis controller. Tracks hormones per session.
pub struct HomeostasisEngine {
    /// Global (non-session) hormone state.
    global: RwLock<HomeostasisState>,
}

struct HomeostasisState {
    hormones: NeuroTransmitters,
    last_update: Instant,
}

impl HomeostasisEngine {
    pub fn new() -> Self {
        Self {
            global: RwLock::new(HomeostasisState {
                hormones: NeuroTransmitters::default(),
                last_update: Instant::now(),
            }),
        }
    }

    /// Get current hormone state after applying time-decay.
    pub fn snapshot(&self) -> NeuroTransmitters {
        let mut state = self.global.write().expect("homeostasis lock poisoned");
        let elapsed = state.last_update.elapsed().as_secs_f32();
        state.hormones.decay(elapsed);
        state.last_update = Instant::now();
        state.hormones.clone()
    }

    /// Process a hormone event.
    pub fn process_event(&self, event: &HormoneEvent) {
        let mut state = self.global.write().expect("homeostasis lock poisoned");
        // Apply decay first.
        let elapsed = state.last_update.elapsed().as_secs_f32();
        state.hormones.decay(elapsed);
        state.last_update = Instant::now();

        // Apply event.
        match event {
            HormoneEvent::RequestStart { complexity } => {
                state.hormones.on_request_start(*complexity);
            }
            HormoneEvent::RequestSuccess => {
                state.hormones.on_request_success();
            }
            HormoneEvent::RequestFailure => {
                state.hormones.on_request_failure();
            }
            HormoneEvent::UserPraise => {
                state.hormones.on_user_praise();
            }
            HormoneEvent::UserFrustration => {
                state.hormones.on_user_frustration();
            }
            HormoneEvent::NovelTask => {
                state.hormones.on_novel_task();
            }
        }
    }
}

/// Events that trigger hormone changes.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum HormoneEvent {
    #[serde(rename = "request_start")]
    RequestStart { complexity: f32 },
    #[serde(rename = "request_success")]
    RequestSuccess,
    #[serde(rename = "request_failure")]
    RequestFailure,
    #[serde(rename = "user_praise")]
    UserPraise,
    #[serde(rename = "user_frustration")]
    UserFrustration,
    #[serde(rename = "novel_task")]
    NovelTask,
}

// ─── Decay Function ─────────────────────────────────────────────────────────

/// Exponential decay toward a setpoint.
///
/// `value` moves toward `setpoint` with the given `half_life` over `elapsed` seconds.
/// After one half-life, the gap shrinks by 50%.
fn decay_toward(value: f32, setpoint: f32, half_life: f32, elapsed: f32) -> f32 {
    if half_life <= 0.0 || elapsed <= 0.0 {
        return value;
    }
    let decay_factor = (0.5_f32).powf(elapsed / half_life);
    setpoint + (value - setpoint) * decay_factor
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mood_is_energized() {
        let nt = NeuroTransmitters::default();
        assert_eq!(nt.mood(), Mood::Energized);
    }

    #[test]
    fn stress_causes_fatigue() {
        let mut nt = NeuroTransmitters::default();
        // Simulate 5 hard requests without recovery.
        for _ in 0..5 {
            nt.on_request_start(0.9);
        }
        assert!(
            nt.cortisol > 0.5,
            "cortisol should be high: {}",
            nt.cortisol
        );
        assert!(nt.energy < 0.5, "energy should be low: {}", nt.energy);
        assert!(
            matches!(nt.mood(), Mood::Fatigued | Mood::Exhausted),
            "mood should be fatigued, got {:?}",
            nt.mood()
        );
    }

    #[test]
    fn praise_boosts_dopamine_and_oxytocin() {
        let mut nt = NeuroTransmitters::default();
        let old_dop = nt.dopamine;
        let old_oxy = nt.oxytocin;
        nt.on_user_praise();
        assert!(nt.dopamine > old_dop);
        assert!(nt.oxytocin > old_oxy);
    }

    #[test]
    fn decay_toward_setpoint_works() {
        // Start at 1.0, setpoint 0.5, half-life 100s, elapsed 100s → ~0.75.
        let result = decay_toward(1.0, 0.5, 100.0, 100.0);
        assert!((result - 0.75).abs() < 0.01, "expected ~0.75, got {result}");
    }

    #[test]
    fn long_absence_reduces_oxytocin() {
        let mut nt = NeuroTransmitters::default();
        nt.oxytocin = 0.80; // High bonding.
        nt.decay(7200.0); // 2 hours of absence (oxytocin HL = 1h).
        // After 2 half-lives: gap = 0.4, factor = 0.25, result = 0.4 + 0.1 = 0.50.
        assert!(nt.oxytocin < 0.55, "oxytocin should decay: {}", nt.oxytocin);
    }

    #[test]
    fn engine_snapshot_applies_decay() {
        let engine = HomeostasisEngine::new();
        engine.process_event(&HormoneEvent::RequestStart { complexity: 0.8 });
        let snap1 = engine.snapshot();
        assert!(snap1.cortisol > SP_CORTISOL);

        // Snapshot again — should have decayed slightly (tiny elapsed).
        let snap2 = engine.snapshot();
        // Cortisol should be same or slightly lower (negligible real-time elapsed).
        assert!(snap2.cortisol <= snap1.cortisol + 0.001);
    }
}
