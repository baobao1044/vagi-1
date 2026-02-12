from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .store import EpisodeStore


@dataclass(slots=True)
class DreamService:
    store: EpisodeStore
    promotion_threshold: float = 0.82
    minimum_pass_rate: float = 0.95

    async def run_once(self, source: str = "manual") -> dict[str, Any]:
        pending = self.store.pending_episodes(limit=500)
        pass_rate = self.store.pass_rate(window=200)
        regression_fail = self.store.regression_fail_count(window=200)
        promoted_ids: list[int] = []

        if pass_rate >= self.minimum_pass_rate and regression_fail == 0:
            for episode in pending:
                if (
                    episode["verifier_pass"] == 1
                    and float(episode["trust_score"]) >= self.promotion_threshold
                ):
                    self.store.promote_episode(int(episode["id"]))
                    promoted_ids.append(int(episode["id"]))

        return {
            "run_id": f"dream-{uuid.uuid4()}",
            "source": source,
            "promoted_count": len(promoted_ids),
            "pass_rate": pass_rate,
            "regression_fail": regression_fail,
            "threshold": self.promotion_threshold,
            "promoted_episode_ids": promoted_ids,
        }


class DreamScheduler:
    def __init__(self, service: DreamService, hour: int, minute: int) -> None:
        self.service = service
        self.hour = hour
        self.minute = minute
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._stop_event.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        await self._task
        self._task = None

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            wait_seconds = _seconds_until(self.hour, self.minute)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_seconds)
                break
            except TimeoutError:
                await self.service.run_once(source="scheduled")


def _seconds_until(hour: int, minute: int) -> float:
    now = datetime.now(UTC)
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()

