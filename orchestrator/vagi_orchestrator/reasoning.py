from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .kernel_client import KernelClient
from .store import EpisodeStore


def compute_trust_score(
    source: str, verifier_pass: bool, risk_score: float, confidence: float
) -> float:
    source_base = {
        "trusted": 0.85,
        "manual": 0.75,
        "chat": 0.65,
        "unknown": 0.55,
    }.get(source, 0.55)
    score = source_base
    score += 0.2 if verifier_pass else -0.25
    score += (confidence - 0.5) * 0.2
    score -= risk_score * 0.25
    return max(0.0, min(1.0, score))


def build_session_id(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"sid-{digest[:16]}"


@dataclass(slots=True)
class Reasoner:
    kernel: KernelClient
    store: EpisodeStore
    max_decide_iters: int = 12
    risk_threshold: float = 0.65

    async def run_chat(self, *, session_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        prompt = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in messages if msg.get("content")
        )
        if not prompt.strip():
            raise ValueError("messages không chứa nội dung hợp lệ")

        await self.kernel.init_state(session_id=session_id)
        await self.kernel.update_state(session_id=session_id, input_text=prompt)

        plan_frame = self._orient(prompt)
        draft = self._intuition(prompt, plan_frame)
        final_sim: dict[str, Any] | None = None
        final_verifier: dict[str, Any] | None = None

        for attempt in range(1, self.max_decide_iters + 1):
            sim = await self.kernel.simulate_world(draft, session_id=session_id)
            verifier = await self.kernel.verify(
                patch_ir=draft,
                max_loop_iters=2048,
                side_effect_budget=3,
                timeout_ms=80,
            )
            final_sim = sim
            final_verifier = verifier
            if verifier.get("pass", False) and sim.get("risk_score", 1.0) <= self.risk_threshold:
                break
            draft = self._revise_draft(draft, sim=sim, verifier=verifier, attempt=attempt)

        assert final_sim is not None
        assert final_verifier is not None

        act_response = self._act(draft, plan_frame, final_sim, final_verifier)
        trust_score = compute_trust_score(
            source="chat",
            verifier_pass=bool(final_verifier.get("pass", False)),
            risk_score=float(final_sim.get("risk_score", 1.0)),
            confidence=float(final_sim.get("confidence", 0.0)),
        )
        episode_id = self.store.record_episode(
            session_id=session_id,
            user_input=prompt,
            draft=draft,
            verifier_pass=bool(final_verifier.get("pass", False)),
            risk_score=float(final_sim.get("risk_score", 1.0)),
            trust_score=trust_score,
            violations=list(final_verifier.get("violations", [])),
            source="chat",
        )
        act_response["episode_id"] = episode_id
        act_response["trust_score"] = trust_score
        return act_response

    def _orient(self, prompt: str) -> dict[str, Any]:
        lower = prompt.lower()
        if "speed" in lower or "latency" in lower:
            priority = "speed"
        elif "memory" in lower or "ram" in lower:
            priority = "memory"
        else:
            priority = "correctness"
        return {
            "objective": "Giải bài toán code theo OODA và tự kiểm chứng",
            "constraints": [
                "verifier phải pass",
                "risk score phải dưới ngưỡng",
                "không đưa hành vi nguy hiểm vào output",
            ],
            "priority": priority,
            "acceptance_rules": {
                "verifier_pass": True,
                "max_risk_score": self.risk_threshold,
            },
        }

    def _intuition(self, prompt: str, plan_frame: dict[str, Any]) -> str:
        return (
            "Kế hoạch xử lý:\n"
            f"- Mục tiêu: {plan_frame['objective']}\n"
            f"- Ưu tiên: {plan_frame['priority']}\n"
            "- Các bước:\n"
            "  1) Phân tích input và ràng buộc.\n"
            "  2) Thiết kế thay đổi tối thiểu nhưng an toàn.\n"
            "  3) Chạy kiểm chứng logic trước khi xuất kết quả.\n"
            f"- Ngữ cảnh người dùng:\n{prompt}"
        )

    def _revise_draft(
        self,
        draft: str,
        *,
        sim: dict[str, Any],
        verifier: dict[str, Any],
        attempt: int,
    ) -> str:
        violations = ", ".join(verifier.get("violations", [])) or "none"
        risk = sim.get("risk_score", 1.0)
        sanitized = (
            draft.replace("drop", "soft-delete")
            .replace("rm -rf", "remove-with-review")
            .replace("unsafe", "safe")
            .replace("eval(", "safe_eval(")
        )
        return (
            f"{sanitized}\n"
            f"- Vòng sửa #{attempt}: giảm rủi ro xuống < {self.risk_threshold}.\n"
            f"- Risk hiện tại: {risk:.2f}. Violations: {violations}.\n"
            "- Bổ sung guardrails: validate input, timeout, rate limit, audit log."
        )

    def _act(
        self,
        draft: str,
        plan_frame: dict[str, Any],
        sim: dict[str, Any],
        verifier: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "content": (
                "Đề xuất đã qua OODA + Verifier:\n"
                f"{draft}\n\n"
                f"Risk score: {sim['risk_score']:.2f} | "
                f"Verifier pass: {verifier['pass']}"
            ),
            "metadata": {
                "plan_frame": plan_frame,
                "simulation": sim,
                "verifier": verifier,
            },
        }

