import asyncio
import time

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import (
    INIT_ANTENNAS_JOINT_POSITIONS,
    INIT_HEAD_POSE,
    SLEEP_ANTENNAS_JOINT_POSITIONS,
    SLEEP_HEAD_POSE,
)


def reset_position(reachy: ReachyMini) -> None:
    reachy.goto_target(
        head=INIT_HEAD_POSE,
        antennas=INIT_ANTENNAS_JOINT_POSITIONS,
        duration=1.0,
    )
    reachy.wake_up()


class ChattingMovements:
    def __init__(self, reachy: ReachyMini) -> None:
        self._reachy = reachy

    def __enter__(self):
        reset_position(self._reachy)

        self._reachy.enable_wobbling()
        self._reachy.start_head_tracking()

    def __exit__(self, exc_type, exc, tb):
        self._reachy.stop_head_tracking()
        self._reachy.disable_wobbling()


class SleepingMovements:
    def __init__(self, reachy: ReachyMini) -> None:
        self._reachy = reachy

        self._task: asyncio.Task | None = None

    def _hide_away(self) -> None:
        current_positions, _ = self._reachy.get_current_joint_positions()
        init_positions = [
            6.959852054044218e-07,
            0.5251518455536499,
            -0.668710345667336,
            0.6067086443974802,
            -0.606711497194891,
            0.6687148024583701,
            -0.5251586523105128,
        ]
        dist = np.linalg.norm(np.array(current_positions) - np.array(init_positions))
        if dist > 0.2:
            self._reachy.goto_target(
                head=INIT_HEAD_POSE,
                antennas=SLEEP_ANTENNAS_JOINT_POSITIONS,
                duration=1.0,
            )
            time.sleep(0.2)

        self._reachy.goto_target(
            head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS, duration=2.0
        )

        self._reachy._last_head_pose = SLEEP_HEAD_POSE

    def _check_world(self) -> None:
        for y in [-0.5, 0.5]:
            self._reachy.look_at_world(x=1.0, y=y, z=0.2, duration=1.0)
        time.sleep(1.0)
        self._reachy.goto_target(
            head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS, duration=1.0
        )

    async def _loop(self) -> None:
        self._hide_away()
        while True:
            await asyncio.sleep(5.0)
            self._check_world()

    def __enter__(self):
        self._task = asyncio.create_task(self._loop())

    def __exit__(self, exc_type, exc, tb):
        if self._task:
            self._task.cancel()
