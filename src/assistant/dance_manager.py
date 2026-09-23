import copy
import json
import math
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import (
    InterpolationTechnique,
    distance_between_poses,
)
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from reachy_mini_dances_library.rhythmic_motion import (
    AVAILABLE_ANTENNA_MOVES,
    MoveOffsets,
)

# ---------------------------------------------------------------------------
# Constants & dance catalog
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
PRESET_FILE = APP_DIR / "dance_params.json"

WAVEFORMS = ["sin", "cos", "square", "triangle", "sawtooth"]
BPM_RANGE = (40.0, 180.0)
AMP_SCALE_RANGE = (0.0, 2.0)
DEFAULT_GLOBALS = {"bpm": 110.0, "amplitude_scale": 1.0}
RAD_NAME_OVERRIDES = {"pitch_amp", "roll_amp", "yaw_amp"}
METER_NAME_OVERRIDES = {"z_amp", "y_amp"}

DANCE_AMPLITUDE_DEFAULTS: dict[str, float] = {
    "chicken_peck": 0.75,
    "chin_lead": 0.8,
    "dizzy_spin": 0.85,
    "grid_snap": 0.5,
    "groovy_sway_and_roll": 0.7,
    "head_tilt_roll": 0.8,
    "headbanger_combo": 0.4,
    "interwoven_spirals": 0.7,
    "jackson_square": 0.6,
    "neck_recoil": 0.8,
    "pendulum_swing": 0.85,
    "polyrhythm_combo": 0.7,
    "sharp_side_tilt": 0.45,
    "side_glance_flick": 0.8,
    "side_peekaboo": 0.7,
    "side_to_side_sway": 0.6,
    "simple_nod": 0.6,
    "stumble_and_recover": 1.0,
    "uh_huh_tilt": 0.8,
    "yeah_nod": 0.85,
}


class ParamSpec(BaseModel):
    name: str
    label: str
    type: str
    value: float | str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    options: list[str] | None = None


class SelectPayload(BaseModel):
    name: str = Field(..., description="Name of the dance to select")


class ParamUpdatePayload(BaseModel):
    name: str
    params: dict[str, float | str]
    apply: bool = False


class GlobalSettingsPayload(BaseModel):
    bpm: float


class TogglePayload(BaseModel):
    playing: bool


class ResetPayload(BaseModel):
    name: str


def _normalize_params(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


DANCE_CATALOG: dict[str, dict[str, Any]] = {
    name: {
        "fn": fn,
        "default_params": _normalize_params(copy.deepcopy(params)),
        "metadata": metadata or {},
        "label": name.replace("_", " ").title(),
    }
    for name, (fn, params, metadata) in AVAILABLE_MOVES.items()
}

DANCE_ORDER = sorted(DANCE_CATALOG.keys())
DEFAULT_DANCE = (
    "groovy_sway_and_roll"
    if "groovy_sway_and_roll" in DANCE_CATALOG
    else DANCE_ORDER[0]
)
ANTENNA_CHOICES = sorted(AVAILABLE_ANTENNA_MOVES.keys())


# ---------------------------------------------------------------------------
# Helper functions (units, persistence, specs)
# ---------------------------------------------------------------------------


def build_factory_param_map() -> dict[str, dict[str, Any]]:
    return {
        name: _normalize_params(copy.deepcopy(DANCE_CATALOG[name]["default_params"]))
        for name in DANCE_ORDER
    }


def build_factory_amplitude_map() -> dict[str, float]:
    defaults = {
        name: DANCE_AMPLITUDE_DEFAULTS.get(name, DEFAULT_GLOBALS["amplitude_scale"])
        for name in DANCE_ORDER
    }
    return defaults


def default_globals_block() -> dict[str, Any]:
    factory_amp = build_factory_amplitude_map()
    return {
        "bpm": DEFAULT_GLOBALS["bpm"],
        "amplitude_scale": DEFAULT_GLOBALS["amplitude_scale"],
        "amplitude_per_dance": {
            "factory": copy.deepcopy(factory_amp),
            "current": copy.deepcopy(factory_amp),
        },
    }


def clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def param_is_degree(name: str) -> bool:
    return "rad" in name or name in RAD_NAME_OVERRIDES


def param_is_meter(name: str) -> bool:
    return (
        name.endswith("_m") or name in METER_NAME_OVERRIDES or name.endswith("_amp_m")
    )


def to_ui_units(name: str, value: float) -> tuple[float, str | None]:
    if param_is_degree(name):
        return math.degrees(value), "deg"
    if param_is_meter(name):
        return value * 1000.0, "mm"
    return value, None


def from_ui_units(name: str, value: float) -> float:
    if param_is_degree(name):
        return math.radians(value)
    if param_is_meter(name):
        return value / 1000.0
    return value


def range_for_param(name: str, ui_value: float) -> tuple[float, float, float]:
    if name == "subcycles_per_beat":
        return 0.1, 4.0, 0.05
    if "phase_offset" in name:
        return 0.0, 1.0, 0.01
    if param_is_degree(name):
        span = min(max(abs(ui_value) * 3.0, 30.0), 180.0)
        return -span, span, 0.5
    if param_is_meter(name):
        span = min(max(abs(ui_value) * 3.0, 30.0), 150.0)
        return -span, span, 1.0
    base = max(abs(ui_value), 1.0)
    span = max(base * 4.0, 2.0)
    step = max(span / 100.0, 0.01)
    return -span, span, step


def build_param_spec(name: str, value: Any) -> ParamSpec:
    label = name.replace("_", " ").title()

    if isinstance(value, str):
        options = WAVEFORMS if name == "waveform" else ANTENNA_CHOICES
        if value not in options:
            value = options[0]
        return ParamSpec(
            name=name,
            label=label,
            type="select",
            value=value,
            options=options,
        )

    if not isinstance(value, (int, float)):
        raise ValueError(f"Unsupported parameter type for {name}: {type(value)}")

    ui_value, unit = to_ui_units(name, float(value))
    min_val, max_val, step = range_for_param(name, ui_value)
    return ParamSpec(
        name=name,
        label=label,
        type="number",
        value=ui_value,
        min=min_val,
        max=max_val,
        step=step,
        unit=unit,
    )


def load_persisted() -> dict[str, Any]:
    factory_param_defaults = build_factory_param_map()
    factory_amp_defaults = build_factory_amplitude_map()
    default_payload = {
        "globals": {
            "bpm": DEFAULT_GLOBALS["bpm"],
            "amplitude_scale": DEFAULT_GLOBALS["amplitude_scale"],
            "amplitude_per_dance": {
                "factory": copy.deepcopy(factory_amp_defaults),
                "current": copy.deepcopy(factory_amp_defaults),
            },
        },
        "dances": {
            "factory": copy.deepcopy(factory_param_defaults),
            "current": copy.deepcopy(factory_param_defaults),
        },
    }

    if not PRESET_FILE.exists():
        return default_payload
    try:
        loaded = json.loads(PRESET_FILE.read_text())
    except Exception:
        return default_payload

    globals_section = loaded.get("globals") or {}
    dances_section = loaded.get("dances") or {}
    legacy_amp = globals_section.get(
        "amplitude_scale", DEFAULT_GLOBALS["amplitude_scale"]
    )

    def sanitize_param_map(
        raw_map: dict[str, Any] | None, fallback: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        result = {name: copy.deepcopy(fallback[name]) for name in DANCE_ORDER}
        if isinstance(raw_map, dict):
            for name, overrides in raw_map.items():
                if name not in result or not isinstance(overrides, dict):
                    continue
                result[name].update(_normalize_params(overrides))
        return result

    def sanitize_amplitude_map(
        raw_map: dict[str, Any] | None, fallback: dict[str, float]
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in DANCE_ORDER:
            base = fallback.get(name, DEFAULT_GLOBALS["amplitude_scale"])
            value = base
            if isinstance(raw_map, dict) and name in raw_map:
                try:
                    value = float(raw_map[name])
                except TypeError, ValueError:
                    value = base
            result[name] = clamp(value, AMP_SCALE_RANGE)
        return result

    factory_params_section = (
        dances_section.get("factory") if isinstance(dances_section, dict) else None
    )
    if factory_params_section is None:
        factory_params = copy.deepcopy(factory_param_defaults)
    else:
        factory_params = sanitize_param_map(
            factory_params_section, factory_param_defaults
        )

    current_params_section: dict[str, Any] | None
    if isinstance(dances_section, dict) and "factory" in dances_section:
        current_params_section = dances_section.get("current")
    else:
        current_params_section = (
            dances_section if isinstance(dances_section, dict) else None
        )

    current_params = sanitize_param_map(current_params_section, factory_params)

    amp_section = globals_section.get("amplitude_per_dance")
    amp_factory_section = None
    amp_current_section = None
    if (
        isinstance(amp_section, dict)
        and "factory" in amp_section
        and "current" in amp_section
    ):
        amp_factory_section = amp_section.get("factory")
        amp_current_section = amp_section.get("current")
    elif isinstance(amp_section, dict):
        amp_current_section = amp_section

    factory_amplitudes = sanitize_amplitude_map(
        amp_factory_section, factory_amp_defaults
    )
    current_amplitudes = sanitize_amplitude_map(amp_current_section, factory_amplitudes)
    if amp_current_section is None:
        current_amplitudes = sanitize_amplitude_map(
            {name: legacy_amp for name in DANCE_ORDER}, factory_amplitudes
        )

    return {
        "globals": {
            "bpm": float(globals_section.get("bpm", DEFAULT_GLOBALS["bpm"])),
            "amplitude_scale": float(legacy_amp),
            "amplitude_per_dance": {
                "factory": factory_amplitudes,
                "current": current_amplitudes,
            },
        },
        "dances": {
            "factory": factory_params,
            "current": current_params,
        },
    }


def serialize_state(snapshot: dict[str, Any]) -> None:
    payload = {
        "globals": {
            "bpm": snapshot["bpm"],
            "amplitude_scale": snapshot["amplitude_per_dance"].get(
                snapshot["selected"], DEFAULT_GLOBALS["amplitude_scale"]
            ),
            "amplitude_per_dance": {
                "factory": snapshot["factory_amplitude_per_dance"],
                "current": snapshot["amplitude_per_dance"],
            },
        },
        "dances": {
            "factory": snapshot["factory_params"],
            "current": snapshot["dance_params"],
        },
    }
    PRESET_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Reachy Mini App
# ---------------------------------------------------------------------------
class Simpledances(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        state_lock = threading.Lock()
        persisted = load_persisted()

        factory_params = {
            name: copy.deepcopy(
                persisted["dances"]["factory"].get(
                    name,
                    _normalize_params(
                        copy.deepcopy(DANCE_CATALOG[name]["default_params"])
                    ),
                )
            )
            for name in DANCE_ORDER
        }

        dance_params = {
            name: copy.deepcopy(
                persisted["dances"]["current"].get(name, factory_params[name])
            )
            for name in DANCE_ORDER
        }

        factory_amplitude_per_dance = {
            name: clamp(
                persisted["globals"]["amplitude_per_dance"]["factory"].get(
                    name,
                    DANCE_AMPLITUDE_DEFAULTS.get(
                        name, DEFAULT_GLOBALS["amplitude_scale"]
                    ),
                ),
                AMP_SCALE_RANGE,
            )
            for name in DANCE_ORDER
        }

        amplitude_per_dance = {
            name: clamp(
                persisted["globals"]["amplitude_per_dance"]["current"].get(
                    name, factory_amplitude_per_dance[name]
                ),
                AMP_SCALE_RANGE,
            )
            for name in DANCE_ORDER
        }

        runtime_state: dict[str, Any] = {
            "selected": DEFAULT_DANCE,
            "playing": True,
            "bpm": clamp(
                persisted["globals"].get("bpm", DEFAULT_GLOBALS["bpm"]), BPM_RANGE
            ),
            "dance_params": dance_params,
            "factory_params": factory_params,
            "amplitude_per_dance": amplitude_per_dance,
            "factory_amplitude_per_dance": factory_amplitude_per_dance,
            "last_offsets": {
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "antennas": [0.0, 0.0],
            },
            "last_saved": None,
            "phase_start": time.perf_counter(),
            "active_bpm": clamp(
                persisted["globals"].get("bpm", DEFAULT_GLOBALS["bpm"]), BPM_RANGE
            ),
            "active_amplitude_per_dance": copy.deepcopy(amplitude_per_dance),
            "active_params": copy.deepcopy(dance_params),
            "returning_home": False,
            "auto_restart_requested": False,
        }

        def persist_locked() -> None:
            serialize_state(runtime_state)
            runtime_state["last_saved"] = datetime.utcnow().isoformat() + "Z"

        def refresh_active_state_locked() -> None:
            runtime_state["active_bpm"] = runtime_state["bpm"]
            runtime_state["active_amplitude_per_dance"] = copy.deepcopy(
                runtime_state["amplitude_per_dance"]
            )
            runtime_state["active_params"] = {
                name: copy.deepcopy(params)
                for name, params in runtime_state["dance_params"].items()
            }
            runtime_state["phase_start"] = time.perf_counter()

        def get_param_specs(dance_name: str) -> list[ParamSpec]:
            amplitude_value = runtime_state["amplitude_per_dance"].get(
                dance_name, DEFAULT_GLOBALS["amplitude_scale"]
            )
            specs: list[ParamSpec] = [
                ParamSpec(
                    name="__amplitude_scale",
                    label="Amplitude Scale",
                    type="number",
                    value=amplitude_value,
                    min=AMP_SCALE_RANGE[0],
                    max=AMP_SCALE_RANGE[1],
                    step=0.05,
                    unit="×",
                )
            ]
            specs.extend(
                build_param_spec(param, value)
                for param, value in runtime_state["dance_params"][dance_name].items()
            )
            return specs

        # -------------------- API Endpoints --------------------
        @self.settings_app.get("/api/dances")
        async def list_dances():
            return {
                "dances": [
                    {
                        "name": name,
                        "label": DANCE_CATALOG[name]["label"],
                        "description": DANCE_CATALOG[name]["metadata"].get(
                            "description", ""
                        ),
                        "default_duration_beats": DANCE_CATALOG[name]["metadata"].get(
                            "default_duration_beats", 4
                        ),
                    }
                    for name in DANCE_ORDER
                ]
            }

        @self.settings_app.get("/api/state")
        async def get_state():
            with state_lock:
                selected = runtime_state["selected"]
                meta = DANCE_CATALOG[selected]["metadata"]
                return {
                    "selected": selected,
                    "playing": runtime_state["playing"],
                    "bpm": runtime_state["bpm"],
                    "amplitude_scale": runtime_state["amplitude_per_dance"].get(
                        selected, DEFAULT_GLOBALS["amplitude_scale"]
                    ),
                    "param_specs": [
                        spec.model_dump() for spec in get_param_specs(selected)
                    ],
                    "description": meta.get("description", ""),
                    "last_saved": runtime_state["last_saved"],
                }

        @self.settings_app.post("/api/select")
        async def select_dance(payload: SelectPayload):
            if payload.name not in runtime_state["dance_params"]:
                raise HTTPException(status_code=404, detail="Unknown dance")
            with state_lock:
                runtime_state["selected"] = payload.name
                runtime_state["phase_start"] = time.perf_counter()
                specs = [spec.model_dump() for spec in get_param_specs(payload.name)]
                meta = DANCE_CATALOG[payload.name]["metadata"]
                return {
                    "selected": payload.name,
                    "param_specs": specs,
                    "description": meta.get("description", ""),
                    "last_saved": runtime_state["last_saved"],
                }

        @self.settings_app.post("/api/params")
        async def update_params(payload: ParamUpdatePayload):
            dance_params = runtime_state["dance_params"].get(payload.name)
            if dance_params is None:
                raise HTTPException(status_code=404, detail="Unknown dance")

            changed = False
            restart_after_apply = False
            with state_lock:
                was_playing = runtime_state["playing"]
                for key, ui_value in payload.params.items():
                    if key == "__amplitude_scale":
                        try:
                            numeric = clamp(float(ui_value), AMP_SCALE_RANGE)
                        except TypeError, ValueError:
                            continue
                        current_amp = runtime_state["amplitude_per_dance"].get(
                            payload.name, DEFAULT_GLOBALS["amplitude_scale"]
                        )
                        if math.isclose(
                            numeric, current_amp, rel_tol=1e-4, abs_tol=1e-4
                        ):
                            continue
                        runtime_state["amplitude_per_dance"][payload.name] = numeric
                        changed = True
                        continue
                    if key not in dance_params:
                        continue
                    current = dance_params[key]
                    if isinstance(current, str):
                        if not isinstance(ui_value, str):
                            continue
                        if key == "waveform" and ui_value not in WAVEFORMS:
                            continue
                        if (
                            key == "antenna_move_name"
                            and ui_value not in ANTENNA_CHOICES
                        ):
                            continue
                        dance_params[key] = ui_value
                        changed = True
                    else:
                        try:
                            numeric = from_ui_units(key, float(ui_value))
                        except TypeError, ValueError:
                            continue
                        if math.isclose(numeric, current, rel_tol=1e-4, abs_tol=1e-4):
                            continue
                        dance_params[key] = numeric
                        changed = True
                specs = [spec.model_dump() for spec in get_param_specs(payload.name)]
                should_restart = payload.apply and was_playing
                if changed:
                    persist_locked()
                if should_restart:
                    runtime_state["playing"] = False
                    runtime_state["phase_start"] = time.perf_counter()
                restart_after_apply = should_restart
            if restart_after_apply:
                start_smooth_return_to_neutral(resume_after=True)
            return {
                "param_specs": specs,
                "last_saved": runtime_state["last_saved"],
                "changed": changed,
                "restart_queued": restart_after_apply,
            }

        @self.settings_app.post("/api/globals")
        async def update_globals(payload: GlobalSettingsPayload):
            with state_lock:
                runtime_state["bpm"] = clamp(float(payload.bpm), BPM_RANGE)
                return {
                    "bpm": runtime_state["bpm"],
                    "amplitude_scale": runtime_state["amplitude_per_dance"].get(
                        runtime_state["selected"], DEFAULT_GLOBALS["amplitude_scale"]
                    ),
                    "last_saved": runtime_state["last_saved"],
                }

        @self.settings_app.post("/api/save")
        async def save_state_endpoint():
            with state_lock:
                persist_locked()
                return {"last_saved": runtime_state["last_saved"]}

        def start_smooth_return_to_neutral(resume_after: bool = False):
            with state_lock:
                if resume_after:
                    runtime_state["auto_restart_requested"] = True
                if runtime_state["returning_home"]:
                    return
                runtime_state["returning_home"] = True

            def _worker():
                try:
                    current_pose = reachy_mini.get_current_head_pose()
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[SimpleDances] Failed to read pose for smooth stop: {exc}")
                    duration = 0.2
                else:
                    _, _, magic_distance = distance_between_poses(
                        current_pose, neutral_pose
                    )
                    duration = max(min(magic_distance * 0.02, 2.0), 0.15)

                try:
                    reachy_mini.goto_target(
                        head=neutral_pose,
                        antennas=neutral_antennas,
                        duration=duration,
                        method=InterpolationTechnique.MIN_JERK,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[SimpleDances] Smooth stop failed: {exc}")
                finally:
                    with state_lock:
                        runtime_state["returning_home"] = False
                        should_resume = runtime_state.get(
                            "auto_restart_requested", False
                        )
                        if should_resume:
                            runtime_state["auto_restart_requested"] = False
                            runtime_state["playing"] = True
                            refresh_active_state_locked()

            threading.Thread(target=_worker, daemon=True).start()

        @self.settings_app.post("/api/toggle")
        async def toggle_playback(payload: TogglePayload):
            start_smooth_stop = False
            with state_lock:
                target_state = bool(payload.playing)
                was_playing = runtime_state["playing"]
                runtime_state["auto_restart_requested"] = False
                runtime_state["playing"] = target_state
                if target_state and not was_playing:
                    refresh_active_state_locked()
                else:
                    runtime_state["phase_start"] = time.perf_counter()
                start_smooth_stop = (not target_state) and was_playing
            if start_smooth_stop:
                start_smooth_return_to_neutral()
            return {"playing": runtime_state["playing"]}

        @self.settings_app.post("/api/reset")
        async def reset_dance(payload: ResetPayload):
            if payload.name not in runtime_state["dance_params"]:
                raise HTTPException(status_code=404, detail="Unknown dance")
            restart_after = False
            with state_lock:
                factory = runtime_state["factory_params"].get(payload.name)
                if factory is None:
                    factory = _normalize_params(
                        copy.deepcopy(DANCE_CATALOG[payload.name]["default_params"])
                    )
                    runtime_state["factory_params"][payload.name] = factory
                runtime_state["dance_params"][payload.name] = copy.deepcopy(factory)
                runtime_state["amplitude_per_dance"][payload.name] = runtime_state[
                    "factory_amplitude_per_dance"
                ].get(payload.name, DEFAULT_GLOBALS["amplitude_scale"])
                was_playing = runtime_state["playing"]
                restart_after = was_playing
                if restart_after:
                    runtime_state["playing"] = False
                    runtime_state["phase_start"] = time.perf_counter()
                    runtime_state["auto_restart_requested"] = True
                specs = [spec.model_dump() for spec in get_param_specs(payload.name)]
                description = DANCE_CATALOG[payload.name]["metadata"].get(
                    "description", ""
                )
            if restart_after:
                start_smooth_return_to_neutral(resume_after=True)
            return {
                "param_specs": specs,
                "description": description,
                "last_saved": runtime_state["last_saved"],
                "changed": True,
                "restart_queued": restart_after,
            }

        @self.settings_app.get("/api/visualization")
        async def visualization_snapshot():
            with state_lock:
                return {
                    "offsets": runtime_state["last_offsets"],
                    "playing": runtime_state["playing"],
                    "bpm": runtime_state["bpm"],
                }

        neutral_pose = create_head_pose()
        neutral_antennas = np.array([-0.1745, 0.1745])
        reachy_mini.set_target(head=neutral_pose, antennas=neutral_antennas)

        loop_dt = 0.02

        while not stop_event.is_set():
            with state_lock:
                selected = runtime_state["selected"]
                playing = runtime_state["playing"]
                params_source = (
                    runtime_state["active_params"]
                    if playing
                    else runtime_state["dance_params"]
                )
                bpm = runtime_state["active_bpm"] if playing else runtime_state["bpm"]
                amp_map = (
                    runtime_state["active_amplitude_per_dance"]
                    if playing
                    else runtime_state["amplitude_per_dance"]
                )
                amp_scale = amp_map.get(selected, DEFAULT_GLOBALS["amplitude_scale"])
                params = copy.deepcopy(params_source[selected])
                phase_start = runtime_state["phase_start"]
                returning_home = runtime_state["returning_home"]

            head_pose = neutral_pose
            antennas_cmd = neutral_antennas
            offsets_payload = {
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0],
                "antennas": [0.0, 0.0],
            }

            if playing:
                fn: Callable[..., MoveOffsets] | None = DANCE_CATALOG[selected]["fn"]
                t_now = time.perf_counter()
                elapsed = max(t_now - phase_start, 0.0)
                beats = elapsed * (bpm / 60.0)
                try:
                    offsets = fn(beats, **params)
                except Exception as exc:  # pragma: no cover - safety net
                    print(f"[SimpleDances] Move error ({selected}): {exc}")
                    pos = np.zeros(3)
                    ori = np.zeros(3)
                    antennas = np.zeros(2)
                else:
                    pos = np.asarray(offsets.position_offset, dtype=float)
                    ori = np.asarray(offsets.orientation_offset, dtype=float)
                    antennas = np.asarray(offsets.antennas_offset, dtype=float)

                pos *= amp_scale
                ori *= amp_scale
                antennas *= amp_scale

                head_pose = create_head_pose(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    roll=float(ori[0]),
                    pitch=float(ori[1]),
                    yaw=float(ori[2]),
                    degrees=False,
                    mm=False,
                )
                antennas_cmd = antennas

                offsets_payload = {
                    "position": pos.tolist(),
                    "orientation": ori.tolist(),
                    "antennas": antennas.tolist(),
                }

            should_send = not (not playing and returning_home)
            if should_send:
                reachy_mini.set_target(head=head_pose, antennas=antennas_cmd)

            with state_lock:
                runtime_state["last_offsets"] = offsets_payload

            time.sleep(loop_dt)


if __name__ == "__main__":
    app = Simpledances()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
