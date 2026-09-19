import asyncio
import logging
import time

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import (
    INIT_ANTENNAS_JOINT_POSITIONS,
    INIT_HEAD_POSE,
    SLEEP_ANTENNAS_JOINT_POSITIONS,
    SLEEP_HEAD_POSE,
)

logger = logging.getLogger(__name__)


def _reset_position(reachy: ReachyMini) -> None:
    reachy.goto_target(
        head=INIT_HEAD_POSE,
        antennas=INIT_ANTENNAS_JOINT_POSITIONS,
        duration=1.0,
    )


class ChattingMovements:
    def __init__(self, reachy: ReachyMini, head_tracking_weight: float) -> None:
        self._reachy = reachy
        self._head_tracking_weight = head_tracking_weight

        self._animation_ticker: int = 0
        self._antenna_task: asyncio.Task | None = None

    def __enter__(self):
        _reset_position(self._reachy)

        self._reachy.enable_wobbling()
        self._reachy.start_head_tracking(weight=self._head_tracking_weight)

        self._antenna_task = asyncio.create_task(self._antenna_loop())

    def __exit__(self, exc_type, exc, tb):
        self._reachy.stop_head_tracking()
        self._reachy.disable_wobbling()
        if self._antenna_task:
            self._antenna_task.cancel()

    async def _antenna_loop(self) -> None:
        while True:
            self._animation_ticker += 1

            # 4x speed for first 1.5 seconds
            if self._animation_ticker <= 600:
                self._animation_ticker += 3

            x = self._animation_ticker % 100
            x = abs(x - 50) / 200
            self._reachy.set_target_antenna_joint_positions([-x, x])
            await asyncio.sleep(0.01)


class SleepingMovements:
    def __init__(self, reachy: ReachyMini) -> None:
        self._reachy = reachy

        self._task: asyncio.Task | None = None

    def _hide_away(self) -> None:
        self._reachy.goto_target(
            head=SLEEP_HEAD_POSE,
            antennas=SLEEP_ANTENNAS_JOINT_POSITIONS,
            duration=1.0,
        )

    def _check_world(self) -> None:
        for y in [-0.5, 0.5]:
            self._reachy.look_at_world(x=1.0, y=y, z=0.2, duration=1.0)
            time.sleep(1.5)

        self._reachy.goto_target(
            head=SLEEP_HEAD_POSE,
            antennas=SLEEP_ANTENNAS_JOINT_POSITIONS,
            duration=1.0,
        )

    async def _loop(self) -> None:
        self._hide_away()
        while True:
            next_peek_delay = np.random.uniform(60.0, 1800.0)  # 1 - 30 mins
            await asyncio.sleep(next_peek_delay)
            self._check_world()

    def __enter__(self):
        self._task = asyncio.create_task(self._loop())

    def __exit__(self, exc_type, exc, tb):
        if self._task:
            self._task.cancel()
