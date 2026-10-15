"""Lossless frame and command encoding used by the frozen LS20 checkpoint."""

import hashlib

import numpy as np


PAD_ID = 0
NO_COORD_ID = 1
ROW_END_ID = 2
OBS_BEGIN_ID = 3
OBS_END_ID = 4
HISTORY_ID = 5
NOT_FINISHED_ID = 6
TERMINAL_ID = 7
COLOR_BASE = 8
ACTION_BASE = 24
COORD_BASE = 32
RUN_BASE = 96
VOCAB_SIZE = 160
HISTORY = 4
HORIZON = 4
TARGET_LENGTH = HORIZON * 3
PREFIX_LENGTH = 1142


def action_id(value):
    value = str(value).split(".")[-1].upper()
    if value == "RESET":
        value = "0"
    if value.startswith("ACTION"):
        value = value.removeprefix("ACTION")
    value = int(value)
    if not 0 <= value < 8:
        raise ValueError(f"Action ID outside 0..7: {value}")
    return value


def command_tokens(command):
    action = action_id(command["id"])
    payload = command.get("data") or {}
    has_coordinates = "x" in payload or "y" in payload
    if has_coordinates:
        x, y = payload["x"], payload["y"]
        if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
            raise ValueError(f"Invalid command coordinates: {payload}")
        return (ACTION_BASE + action, COORD_BASE + x, COORD_BASE + y)
    if action == 6:
        raise ValueError("Click action has no coordinates")
    return (ACTION_BASE + action, NO_COORD_ID, NO_COORD_ID)


def decode_command(tokens):
    action, x, y = map(int, tokens)
    return {"action_id": action - ACTION_BASE,
            "x": None if x == NO_COORD_ID else x - COORD_BASE,
            "y": None if y == NO_COORD_ID else y - COORD_BASE}


def encode_frame(frame):
    frame = np.asarray(frame)
    if frame.shape != (64, 64) or frame.dtype.kind not in "iu" or frame.min() < 0 or frame.max() > 15:
        raise ValueError(f"Expected integer 64x64 frame with palette 0..15, got {frame.shape}/{frame.dtype}")
    encoded = []
    for row in frame:
        start = 0
        for end in range(1, 65):
            if end == 64 or row[end] != row[start]:
                encoded.extend((COLOR_BASE + int(row[start]), RUN_BASE + end - start - 1))
                start = end
        encoded.append(ROW_END_ID)
    if not np.array_equal(decode_frame(encoded), frame):
        raise AssertionError("Lossless frame RLE did not round-trip")
    return tuple(encoded), hashlib.sha256(frame.astype(np.uint8).tobytes()).hexdigest()


def decode_frame(tokens):
    rows, row = [], []
    cursor = 0
    while cursor < len(tokens):
        color = int(tokens[cursor])
        cursor += 1
        if color == ROW_END_ID:
            if len(row) != 64:
                raise ValueError("RLE row length differs from 64")
            rows.append(row)
            row = []
            continue
        if not COLOR_BASE <= color < ACTION_BASE or cursor == len(tokens):
            raise ValueError("Invalid RLE color or missing run length")
        run = int(tokens[cursor])
        cursor += 1
        if not RUN_BASE <= run < VOCAB_SIZE:
            raise ValueError("Invalid RLE run length")
        row.extend([color - COLOR_BASE] * (run - RUN_BASE + 1))
    if row or len(rows) != 64:
        raise ValueError("RLE frame does not contain exactly 64 complete rows")
    return np.asarray(rows, dtype=np.uint8)


def encode_prefix(observation, history):
    available = sorted({int(action) for action in observation.available_actions})
    level = int(observation.levels_completed)
    if not 0 <= level < 64:
        raise ValueError("level_outside_frozen_vocabulary")
    if not available or len(available) > 8 or any(not 0 <= action < 8 for action in available):
        raise ValueError("invalid_available_actions")
    rle, frame_sha256 = encode_frame(observation.frame[-1])
    previous = [[0, 0, 0]] * max(0, HISTORY - len(history)) + list(history[-HISTORY:])
    prefix = [NOT_FINISHED_ID, COORD_BASE + level, RUN_BASE + len(available) - 1,
              *[ACTION_BASE + action for action in available], HISTORY_ID,
              *[token for packet in previous for token in packet], OBS_BEGIN_ID, *rle, OBS_END_ID]
    info = {"prefix_length": len(prefix), "capacity": PREFIX_LENGTH,
            "prefix_sha256": hashlib.sha256(np.asarray(prefix, dtype="<u2").tobytes()).hexdigest(),
            "frame_sha256": frame_sha256, "history": previous}
    if len(prefix) > PREFIX_LENGTH:
        return None, {**info, "type": "policy_failure", "reason": "prefix_capacity_overflow"}
    return prefix + [PAD_ID] * (PREFIX_LENGTH - len(prefix)), info


def decode_prediction(tokens, available):
    if (not isinstance(tokens, list) or len(tokens) != TARGET_LENGTH or
            any(type(token) is not int for token in tokens)):
        return {"type": "policy_failure", "reason": "malformed_prediction_length_or_type"}
    action_token, x, y = tokens[:3]
    if not ACTION_BASE <= action_token < COORD_BASE:
        return {"type": "policy_failure", "reason": "malformed_action_token"}
    action = action_token - ACTION_BASE
    if action == 6:
        if not COORD_BASE <= x < RUN_BASE or not COORD_BASE <= y < RUN_BASE:
            return {"type": "policy_failure", "reason": "malformed_click_coordinates"}
        payload = {"x": x - COORD_BASE, "y": y - COORD_BASE}
    else:
        if x != NO_COORD_ID or y != NO_COORD_ID:
            return {"type": "policy_failure", "reason": "coordinates_on_nonclick_action"}
        payload = {}
    if action not in available:
        return {"type": "policy_failure", "reason": "unavailable_action", "action_id": action}
    return {"type": "action", "action_id": action, "data": payload}
