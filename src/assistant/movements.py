import asyncio
import logging
import time
from typing import Literal

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import (
    INIT_ANTENNAS_JOINT_POSITIONS,
    INIT_HEAD_POSE,
    SLEEP_ANTENNAS_JOINT_POSITIONS,
    SLEEP_HEAD_POSE,
)
from reachy_mini.utils import create_head_pose
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES

logger = logging.getLogger(__name__)


class MovementManager:
    def __init__(
        self,
        reachy: ReachyMini,
        listen_for_music: bool = True,
    ) -> None:
        self._reachy = reachy
        self._listen_for_music = listen_for_music

        self._anim_loop_task: asyncio.Task | None = None

        self._state: Literal["none", "chatting", "sleeping", "dancing"] = "none"
        self._dance_bpm: float | None = None

        self.start_sleeping()

    def _pop_head_out(self) -> None:
        self._reachy.goto_target(
            head=INIT_HEAD_POSE,
            antennas=INIT_ANTENNAS_JOINT_POSITIONS,
            duration=1.0,
        )

    def reset_state(self) -> None:
        if self._anim_loop_task:
            self._anim_loop_task.cancel()

        self._reachy.stop_head_tracking()
        self._reachy.disable_wobbling()

        self._state = "none"
        self._dance_bpm = None

    def start_chatting(self, *, head_tracking_weight: float = 1.0) -> None:
        self.reset_state()
        self._state = "chatting"

        self._pop_head_out()

        self._reachy.enable_wobbling()
        self._reachy.start_head_tracking(weight=head_tracking_weight)
        self._anim_loop_task = asyncio.create_task(self._antenna_wiggle_loop())

    async def _antenna_wiggle_loop(self) -> None:
        ticker = 0

        try:
            while True:
                ticker += 1

                # 4x speed for first 1.5 seconds
                if ticker <= 600:
                    ticker += 3

                x = ticker % 100
                x = abs(x - 50) / 200
                self._reachy.set_target_antenna_joint_positions([-x, x])
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    def start_sleeping(self) -> None:
        self.reset_state()
        self._state = "sleeping"

        self._anim_loop_task = asyncio.create_task(self._sleep_loop())

    async def _sleep_loop(self) -> None:

        try:
            while True:
                # hide away
                self._reachy.goto_target(
                    head=SLEEP_HEAD_POSE,
                    antennas=SLEEP_ANTENNAS_JOINT_POSITIONS,
                    duration=1.0,
                )

                next_peek_delay = np.random.uniform(120.0, 1200.0)  # 2 - 20 mins
                await asyncio.sleep(next_peek_delay)

                # check world
                head_directions: list[dict[str, float]] = [
                    {"x": 1.0, "y": 0.2, "z": 0.0},  # down + cw
                    {"x": 1.0, "y": -0.3, "z": 0.2},  # up + ccw
                ]
                for dir in head_directions:
                    head_pose = self._reachy.look_at_world(
                        **dir, perform_movement=False
                    )
                    self._reachy.goto_target(
                        head=head_pose, antennas=[-2.4, 2.4], duration=1.0
                    )
                    await asyncio.sleep(1.5)

        except asyncio.CancelledError:
            pass

    def start_dancing(self, *, bpm: int) -> None:
        if self._state != "dancing":
            self.reset_state()
            self._state = "dancing"
            self._pop_head_out()
            self._dance_bpm = bpm
            self._anim_loop_task = asyncio.create_task(self._dance_loop())
            logger.info("started dancing")
        else:
            self._dance_bpm = bpm
            logger.info(f"updated {bpm=}")

    async def _dance_loop(self) -> None:
        beat_counter = 0
        last_time = time.monotonic()

        while True:
            if self._dance_bpm is None:
                break

            dt = time.monotonic() - last_time
            beat_counter += dt * (self._dance_bpm / 60.0)
            last_time = time.monotonic()

            move_fn, runtime_params, _ = AVAILABLE_MOVES["head_tilt_roll"]
            move_offsets = move_fn(beat_counter, **runtime_params)
            head_pose = create_head_pose(
                *move_offsets.position_offset,
                *move_offsets.orientation_offset,
                degrees=False,
            )

            # use antenna position as body yaw so we don't have to re-calculate it
            body_yaw = move_offsets.antennas_offset[0] / 2.0

            self._reachy.set_target(
                head=head_pose,
                antennas=move_offsets.antennas_offset / 3.0,
                body_yaw=body_yaw,
            )

            await asyncio.sleep(0.01)
