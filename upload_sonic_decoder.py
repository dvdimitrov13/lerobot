#!/usr/bin/env python3
"""Provision the SONIC decoder checkpoint at ``lerobot/sonic_decoder``.

Takes NVIDIA's ``nvidia/GEAR-SONIC/model_decoder.onnx``, embeds the SONIC deploy constants
(``kp``/``kd`` PD gains, ``default_angles`` standing pose, the residual ``action_scale``, and
the ``neutral_token`` idle latent) into the ONNX ``metadata_props`` (the convention Holosoma
uses for its gains), and pushes the result to ``lerobot/sonic_decoder``. After this runs, the
runtime loads the decoder *and* every one of these constants straight from the checkpoint --
no motor-physics math at deploy time, so ``sonic_whole_body.py`` carries none of the
armature/bandwidth machinery nor any hardcoded deploy constants.

The constants here are derived once from Unitree motor physics (armature + target bandwidth).
That derivation is intentionally kept in this one-off provisioning script (not the runtime);
the shared/harmonic helper is a separate PR.

Build only (no network/auth needed if the source ONNX is already cached):
    python upload_sonic_decoder.py --out ./sonic_decoder

Build + upload:
    huggingface-cli login          # or export HF_TOKEN=...
    python upload_sonic_decoder.py --upload
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import onnx
from huggingface_hub import hf_hub_download

SRC_REPO_ID = "nvidia/GEAR-SONIC"
SRC_FILENAME = "model_decoder.onnx"
DST_REPO_ID = "lerobot/sonic_decoder"

# ── SONIC deploy-constant derivation (provisioning-time only) ─────────────────
# All constants are (29,) in IsaacLab joint order: legs, waist, arms.
# kp = armature * w**2, kd = 4 * armature * w, with a x2 factor on the stiff joints
# (ankles + waist). action_scale = 0.25 * effort / (armature * w**2) is the residual
# scaling that maps decoder output to a joint-angle delta on top of default_angles.
NATURAL_FREQ = 10.0 * 2.0 * np.pi
MOTOR_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
MOTOR_MODELS = (
    ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2
    + ["7520_14", "5020", "5020"]
    + ["5020", "5020", "5020", "5020", "5020", "4010", "4010"] * 2
)
DOUBLE_INDICES = {4, 5, 10, 11, 13, 14}  # ankles + waist

# Nominal standing pose (rad), 29 joints in IsaacLab order. Decoder actions are residuals
# added on top of this.
DEFAULT_ANGLES = [
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,  # left leg
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,  # right leg
    0.0,
    0.0,
    0.0,  # waist
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,  # left arm
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,  # right arm
]

# Neutral idle token (64-D), held until the first real token arrives. Captured from the
# encoder while the robot stood idle in sim: the encoder is an FSQ bottleneck (~5 bit/dim,
# Div(16)), so tokens live on the 1/16 grid. We store the integer FSQ codes and rescale by
# 1/16 -> an exact on-grid token that decodes to a stable, natural standing pose (unlike the
# literal all-zero token, which is off-manifold and decodes to a slightly goofy stance).
NEUTRAL_TOKEN_CODES = [
    -1,
    3,
    1,
    -1,
    1,
    -3,
    6,
    1,
    1,
    1,
    -2,
    -4,
    -2,
    0,
    -3,
    -1,
    2,
    -1,
    -3,
    -5,
    3,
    1,
    1,
    -4,
    -1,
    -1,
    1,
    -7,
    0,
    1,
    2,
    -2,
    5,
    -2,
    -2,
    -4,
    0,
    -1,
    3,
    -1,
    0,
    -5,
    -1,
    0,
    -4,
    0,
    0,
    -1,
    -1,
    2,
    -2,
    1,
    3,
    3,
    1,
    0,
    0,
    6,
    0,
    -7,
    3,
    0,
    2,
    -2,
]


def compute_kp_kd() -> tuple[list[float], list[float]]:
    """Return (kp, kd) as plain float lists, (29,) in IsaacLab joint order."""

    def stiffness(k):
        return MOTOR_ARMATURE[k] * NATURAL_FREQ**2

    def damping(k):
        return 4.0 * MOTOR_ARMATURE[k] * NATURAL_FREQ

    kp = [(2 if i in DOUBLE_INDICES else 1) * stiffness(k) for i, k in enumerate(MOTOR_MODELS)]
    kd = [(2 if i in DOUBLE_INDICES else 1) * damping(k) for i, k in enumerate(MOTOR_MODELS)]
    return kp, kd


def compute_action_scale() -> list[float]:
    """Return the per-joint residual action scale, (29,) in IsaacLab joint order."""
    return [0.25 * EFFORT[k] / (MOTOR_ARMATURE[k] * NATURAL_FREQ**2) for k in MOTOR_MODELS]


def build(out_dir: pathlib.Path) -> pathlib.Path:
    """Download the source decoder, embed the deploy-constant metadata, save to ``out_dir``."""
    src = hf_hub_download(repo_id=SRC_REPO_ID, filename=SRC_FILENAME)
    model = onnx.load(src)

    kp, kd = compute_kp_kd()
    neutral_token = [c / 16.0 for c in NEUTRAL_TOKEN_CODES]  # FSQ Div(16): codes -> on-grid token
    meta = {prop.key: prop.value for prop in model.metadata_props}
    meta["kp"] = json.dumps(kp)
    meta["kd"] = json.dumps(kd)
    meta["action_scale"] = json.dumps(compute_action_scale())
    meta["default_angles"] = json.dumps(DEFAULT_ANGLES)
    meta["neutral_token"] = json.dumps(neutral_token)
    # Rewrite metadata_props with the merged dict.
    del model.metadata_props[:]
    for key, value in meta.items():
        model.metadata_props.add(key=key, value=value)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / SRC_FILENAME
    onnx.save(model, out_path)
    print(f"Wrote {out_path} with kp/kd/action_scale/default_angles/neutral_token metadata.")
    return out_path


def upload(out_path: pathlib.Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=DST_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=SRC_FILENAME,
        repo_id=DST_REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded {out_path.name} -> {DST_REPO_ID}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("./sonic_decoder"))
    p.add_argument("--upload", action="store_true", help="Push the built ONNX to the hub")
    args = p.parse_args()

    out_path = build(args.out)
    if args.upload:
        upload(out_path)


if __name__ == "__main__":
    main()
