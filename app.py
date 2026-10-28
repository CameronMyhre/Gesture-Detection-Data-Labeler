"""
Developer GUI for frame-by-frame gesture dataset labeling.

Stores raw YOLO pose detections and processed GRU features in
separate, frame-indexed NumPy archives beside the source video.
"""

import asyncio
import base64
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import ast

from nicegui import app, ui
from ultralytics import YOLO
from ultralytics.engine.results import Keypoints
from nicegui import app, events, ui
from gesture_classification_utils import (
    process_keypoints_for_classification,
)
import gesture_classification_utils

# ============================================================
# 1. Configuration
# ============================================================

VIDEO_PATH = Path("2026-10-17_14-33-24_002.mp4")
MODEL_PATH = "yolo26x-pose.pt"

RAW_PATH = VIDEO_PATH.with_name(VIDEO_PATH.stem + "_raw.npz")
GRU_PATH = VIDEO_PATH.with_name(VIDEO_PATH.stem + "_gru.npz")

LABELS = {
    -1: "UNLABELED",
    0: "NONE",
    1: "FOLLOW",
    2: "STAY",
    3: "FETCH",
    4: "COME",
    5: "GIVE",
    6: "INVALID",
}

UNPROCESSED = 0
VALID = 1
INVALID = 2

FEATURE_COUNT = 44
KEYPOINT_COUNT = 17

# Connections between keypoints using YOLO's COCO layout.
SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

confidence = 0.2
used_imgsz = [800, 1280]
auto_labeling = False

# ============================================================
# 2. Camera and model
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames <= 0:
    raise RuntimeError("Video has no readable frames.")

model = YOLO(MODEL_PATH)


# ============================================================
# 3. Frame-indexed storage
# ============================================================

# All arrays use the same frame indexing.

raw_keypoints = np.zeros(
    (total_frames, KEYPOINT_COUNT, 3),
    dtype=np.float32,
)

features = np.zeros(
    (total_frames, FEATURE_COUNT),
    dtype=np.float32,
)

normalized = None

labels = np.full(
    total_frames,
    -1,
    dtype=np.int8,
)

status = np.full(
    total_frames,
    UNPROCESSED,
    dtype=np.uint8,
)

frame_indices = np.arange(total_frames, dtype=np.int32)


def save_npz(path: Path, **arrays):
    """Atomically replace one NumPy archive."""
    temporary_path = path.with_name(path.name + ".tmp")

    try:
        with open(temporary_path, "wb") as file:
            np.savez_compressed(file, **arrays)

        os.replace(temporary_path, path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_dataset():
    """Save raw keypoints, training features, and label data."""

    metadata = {
        "video_name": np.array(VIDEO_PATH.name),
        "frame_indices": frame_indices,
    }

    save_npz(
        RAW_PATH,
        **metadata,
        raw_keypoints=raw_keypoints,
        labels=labels,
        status=status,
    )

    save_npz(
        GRU_PATH,
        **metadata,
        features=features,
        labels=labels,
        status=status,
    )


def load_dataset():
    """Restore existing dataset arrays, validating their shapes."""

    if not RAW_PATH.exists():
        if GRU_PATH.exists():
            raise RuntimeError(
                "GRU archive exists without its raw archive. "
                "Restore the raw archive before continuing."
            )
        return

    with np.load(RAW_PATH, allow_pickle=False) as data:
        if str(data["video_name"]) != VIDEO_PATH.name:
            raise ValueError("Raw archive belongs to another video.")

        if not np.array_equal(data["frame_indices"], frame_indices):
            raise ValueError("Raw archive has incompatible frame indices.")

        if data["raw_keypoints"].shape != raw_keypoints.shape:
            raise ValueError("Raw keypoint array has the wrong shape.")

        if data["labels"].shape != labels.shape:
            raise ValueError("Label array has the wrong shape.")

        if data["status"].shape != status.shape:
            raise ValueError("Status array has the wrong shape.")

        raw_keypoints[:] = data["raw_keypoints"]
        labels[:] = data["labels"]
        status[:] = data["status"]

    # Load the GRU features if the processed archive exists.
    if GRU_PATH.exists():
        with np.load(GRU_PATH, allow_pickle=False) as data:
            if str(data["video_name"]) != VIDEO_PATH.name:
                raise ValueError("GRU archive belongs to another video.")

            if not np.array_equal(data["frame_indices"], frame_indices):
                raise ValueError("GRU archive has incompatible indices.")

            if data["features"].shape != features.shape:
                raise ValueError("GRU feature array has the wrong shape.")

            if not np.array_equal(data["labels"], labels):
                raise ValueError(
                    "Label archives disagree. Restore a consistent "
                    "pair of files before continuing."
                )

            if not np.array_equal(data["status"], status):
                raise ValueError("Processing-status archives disagree.")

            features[:] = data["features"]

    else:
        # Reconstruct a missing GRU archive from saved raw poses.
        for i in np.flatnonzero(status == VALID):
            processed = process_raw_keypoints(raw_keypoints[i])

            if processed is None:
                status[i] = INVALID
                labels[i] = 6
            else:
                features[i] = processed

        save_dataset()


# ============================================================
# 4. Keypoint processing
# ============================================================


def process_raw_keypoints(points: np.ndarray):
    """Convert a stored (17, 3) pose into 44 GRU features."""

    global normalized
    keypoints = Keypoints(
        torch.from_numpy(points.copy()[None, ...]),
        orig_shape=(video_height, video_width),
    )

    processed = process_keypoints_for_classification(keypoints)
    normalized = gesture_classification_utils.normalize_figure(keypoints)

    if processed is None:
        return None

    processed = np.asarray(processed, dtype=np.float32)

    if processed.shape != (FEATURE_COUNT,):
        return None

    if not np.all(np.isfinite(processed)):
        return None

    return processed


def extract_pose(result):
    """Select the highest-confidence detected person."""

    if result.keypoints is None:
        return None

    data = result.keypoints.data

    if data is None or len(data) == 0:
        return None

    # Prefer the highest-confidence bounding box.
    if result.boxes is not None and len(result.boxes) == len(data):
        scores = result.boxes.conf.cpu().numpy()
        person_index = int(np.argmax(scores))
    else:
        person_index = 0

    points = data[person_index].cpu().numpy()

    if points.shape != (KEYPOINT_COUNT, 3):
        return None

    if not np.all(np.isfinite(points)):
        return None

    return points.astype(np.float32)


def mark_invalid(frame_index: int):
    """Mark a frame invalid and zero its training features."""

    status[frame_index] = INVALID
    labels[frame_index] = 6

    features[frame_index].fill(0)


# ============================================================
# 5. Video and visualization
# ============================================================

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

current_frame = 0
current_image = None
updating_slider = False
predicting = False


def frame_to_img(frame: np.ndarray):
    """Convert a BGR frame to an HTML-compatible JPEG."""

    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        raise ValueError("Could not encode frame.")

    encoded = base64.b64encode(buffer).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def draw_keypoints(frame: np.ndarray, points: np.ndarray):
    """Draw saved keypoints on a copy of the original frame."""

    frame = frame.copy()

    # Connect sufficiently confident joints.
    for a, b in SKELETON:
        if points[a, 2] < 0.25 or points[b, 2] < 0.25:
            continue

        p1 = tuple(points[a, :2].astype(int))
        p2 = tuple(points[b, :2].astype(int))

        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    # Draw the individual joints.
    for x, y, confidence in points:
        if confidence < 0.25:
            continue

        cv2.circle(
            frame,
            (int(x), int(y)),
            4,
            (0, 0, 255),
            -1,
        )

    return frame


def update_progress():
    """Refresh dataset status information."""

    processed_count = int(np.count_nonzero(status != UNPROCESSED))

    labeled_count = int(np.count_nonzero(labels != -1))

    pending_count = int(np.count_nonzero(labels == -1))

    progress_label.set_text(
        f"Processed: {processed_count}/{total_frames} | "
        f"Labeled: {labeled_count}/{total_frames} | "
        f"Unlabeled: {pending_count}"
    )


def refresh_display():
    """Display the current frame and its saved metadata."""

    if current_image is None:
        return

    frame = current_image.copy()

    if status[current_frame] != UNPROCESSED:
        frame = draw_keypoints(
            frame,
            raw_keypoints[current_frame],
        )

    image_element.set_source(frame_to_img(frame))

    slider_label.set_text(f"Frame {current_frame} / {total_frames - 1}")

    current_label.set_text(f"Gesture: {LABELS[int(labels[current_frame])]}")

    processing_label.set_text(
        "Pose: "
        + {
            UNPROCESSED: "Not processed",
            VALID: "Valid",
            INVALID: "Invalid",
        }[int(status[current_frame])]
    )

    update_progress()


def show_frame(frame_index, sync_slider=True):
    """Seek to and display an original video frame."""

    global current_frame, current_image, updating_slider

    frame_index = max(
        0,
        min(int(frame_index), total_frames - 1),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()

    if not ret:
        ui.notify("Could not read frame.", type="warning")
        return

    current_frame = frame_index
    current_image = frame

    refresh_display()

    if sync_slider:
        updating_slider = True

        try:
            slider.set_value(current_frame)
        finally:
            updating_slider = False


# ============================================================
# 6. GUI interaction logic
# ============================================================


async def infer_keypoints():
    """Run YOLO and persist the resulting pose features."""

    global predicting, confidence, used_imgsz

    if predicting or current_image is None:
        return

    # Capture the frame index so navigation during inference
    # cannot accidentally write results to another frame.
    frame_index = current_frame
    frame_image = current_image.copy()

    # Reuse existing detections unless explicitly reprocessing.
    if status[frame_index] != UNPROCESSED:
        ui.notify("This frame has already been processed.")
        return

    predicting = True
    predict_button.disable()

    try:
        # Offload inference so the UI remains responsive.
        results = await asyncio.to_thread(
            model.track,
            frame_image,
            verbose=False,
            conf=confidence,
            task="pose",
            imgsz=used_imgsz,
            device=0,
        )

        points = extract_pose(results[0])

        if points is None:
            raw_keypoints[frame_index].fill(0)
            mark_invalid(frame_index)

        else:
            raw_keypoints[frame_index] = points

            processed = process_raw_keypoints(points)

            if processed is None:
                mark_invalid(frame_index)
            else:
                global normalized
                features[frame_index] = processed
                status[frame_index] = VALID

                # Preserve a gesture label if one was assigned.
                # Reset an old INVALID label after a successful retry.
                if labels[frame_index] == 6:
                    labels[frame_index] = -1

        save_dataset()

        if frame_index == current_frame:
            refresh_display()

        ui.notify(f"Processed frame {frame_index}")

    except Exception as error:
        ui.notify(f"Inference error: {error}", type="negative")

    finally:
        predicting = False
        predict_button.enable()


def assign_label(label_id: int):
    """Assign a gesture label to the current video frame."""

    # Invalid frames cannot receive a normal gesture label
    # until inference succeeds.
    if status[current_frame] == INVALID and label_id != 6:
        ui.notify(
            "Reprocess this invalid frame before labeling it.",
            type="warning",
        )
        return

    labels[current_frame] = label_id

    if label_id == 6:
        mark_invalid(current_frame)

    save_dataset()
    refresh_display()


def reprocess_frame():
    """Clear a frame's pose data so YOLO can run again."""

    raw_keypoints[current_frame].fill(0)
    features[current_frame].fill(0)

    status[current_frame] = UNPROCESSED

    if labels[current_frame] == 6:
        labels[current_frame] = -1

    save_dataset()
    refresh_display()


def next_frame():
    show_frame(current_frame + 1)


def previous_frame():
    show_frame(current_frame - 1)


def next_unlabeled():
    """Jump to the next frame without an assigned label."""

    pending = np.flatnonzero(labels == -1)
    later = pending[pending > current_frame]

    if len(later):
        show_frame(int(later[0]))
    elif len(pending):
        show_frame(int(pending[0]))
    else:
        ui.notify("All frames have labels!")


def next_unprocessed():
    """Jump to a frame still needing pose inference."""

    pending = np.flatnonzero(status == UNPROCESSED)
    later = pending[pending > current_frame]

    if len(later):
        show_frame(int(later[0]))
    elif len(pending):
        show_frame(int(pending[0]))
    else:
        ui.notify("All frames have been processed!")


def update_slider(e):
    if not updating_slider:
        show_frame(e.value, sync_slider=False)


def update_number(e):

    global confidence

    confidence = e.value
    ui.notify(f"Confidence changed to: {e.value}")


def update_imgsz(e):

    global used_imgsz

    used_imgsz = ast.literal_eval(e.value)
    ui.notify(f"Updated imgsz to: {e.value}")


async def process_all():
    global auto_labeling, current_frame, total_frames

    # Toggle auto labeling.
    auto_labeling = not auto_labeling

    while auto_labeling:
        await infer_keypoints()
        if current_frame >= total_frames - 1:
            break
        next_frame()


async def handle_key(e: events.KeyEventArguments):
    """Handle keyboard shortcuts for navigation and labeling."""

    # Only respond when a key is initially pressed.
    if not e.action.keydown or e.action.repeat:
        return

    # Ignore keyboard shortcuts involving modifiers.
    if e.modifiers.ctrl or e.modifiers.alt or e.modifiers.meta:
        return

    # Frame navigation.
    if e.key.arrow_right:
        next_frame()
        return

    if e.key.arrow_left:
        previous_frame()
        return

    if e.key == "p":
        await infer_keypoints()
        return

    if e.key == "r":
        reprocess_frame()

    if e.key == "f":
        next_unlabeled()

    if e.key == "g":
        next_unprocessed()

    if e.key.delete:

        data = np.load("2026-10-17_15-12-24_009_gru.npz")
        print(data)
        print("Features:", data["status"][225])

    # Gesture labeling (keys 1-7).
    if e.key.number is not None and 1 <= e.key.number <= 7:
        assign_label(e.key.number - 1)
        next_frame()


def classical_classify(e):

    global current_frame, raw_keypoints, normalized

    if normalized is None:
        ui.notify("Error! Cannot classify empty frame data!")
        return

    # Label the frame.
    label = gesture_classification_utils.classical_classify(normalized)
    ui.notify(f"Classified frame as {label}")
    assign_label(label)


# ============================================================
# 7. Load saved files before creating the interface
# =======================Q=====================================

load_dataset()


# ============================================================
# 8. NiceGUI layout
# ============================================================

ui.dark_mode().enable()

with ui.row().classes("w-full"):

    # Video and frame controls.
    with ui.column().classes("flex-1 min-w-0"):

        placeholder = np.zeros(
            (video_height, video_width, 3),
            dtype=np.uint8,
        )

        image_element = ui.interactive_image(frame_to_img(placeholder)).classes(
            "max-w-full max-h-[75vh] object-contain"
        )

        with ui.row().classes("w-full items-center"):

            slider_label = ui.label("Frame: 0")

            ui.button(
                "Back",
                color="red",
                on_click=previous_frame,
            )

            ui.button(
                "Next",
                color="green",
                on_click=next_frame,
            )

            predict_button = ui.button(
                "Predict Keypoints (p)",
                color="orange",
                on_click=infer_keypoints,
            )

            ui.button(
                "Reprocess (r)",
                color="purple",
                on_click=reprocess_frame,
            )

            ui.button(
                "Add all keypoints",
                on_click=process_all,
            )

            ui.button(
                "Label Frame",
                on_click=classical_classify,
            )

            ui.number(
                label="Confidence",
                value=confidence,
                min=0,
                max=1,
                step=0.01,
                on_change=update_number,
            ).classes("w-20 h-0.9")

            ui.input(label="imgsz (h,w):", value="[800,1280]", on_change=update_imgsz)

        slider = ui.slider(
            min=0,
            max=total_frames - 1,
            step=1,
            value=0,
            on_change=update_slider,
        ).classes("w-full")

        with ui.row():
            ui.button(
                "Next Unlabeled (f)",
                on_click=next_unlabeled,
            )

            ui.button(
                "Next Unprocessed (g)",
                on_click=next_unprocessed,
            )

        progress_label = ui.label("")

    # Gesture labeling controls.
    with ui.column().classes("w-72"):

        ui.label("Gesture Labels").classes("text-h4")

        current_label = ui.label("Gesture: UNLABELED")
        processing_label = ui.label("Pose: Not processed")

        with ui.row():
            ui.button(
                "NONE (1)",
                color="red",
                on_click=lambda: assign_label(0),
            )

            ui.button(
                "FOLLOW (2)",
                color="green",
                on_click=lambda: assign_label(1),
            )

            ui.button(
                "STAY (3)",
                color="pink",
                on_click=lambda: assign_label(2),
            )

            ui.button(
                "FETCH (4)",
                color="blue",
                on_click=lambda: assign_label(3),
            )

        with ui.row():
            ui.button(
                "COME (5)",
                color="orange",
                on_click=lambda: assign_label(4),
            )

            ui.button(
                "GIVE (6)",
                color="purple",
                on_click=lambda: assign_label(5),
            )

            ui.button(
                "INVALID (7)",
                color="red",
                on_click=lambda: assign_label(6),
            )

        ui.label("Gesture Key").classes("text-h5")

        with ui.list():
            ui.item("Follow: Y Pose")
            ui.item("Stay: T-Pose")
            ui.item("Fetch: Right 90, Left -45")
            ui.item("Come: Right 0")
            ui.item("Give: Left 0")

        ui.button(
            "Save Dataset",
            on_click=lambda: (
                save_dataset(),
                ui.notify("Dataset saved."),
            ),
        )
        # ui.element('iframe').props('src="http://127.0.0.1:8080/"').classes('w-full h-[80vh]')


# ============================================================
# 9. Startup / shutdown
# ============================================================

show_frame(0)

app.on_shutdown(cap.release)

ui.keyboard(
    on_key=handle_key,
    repeating=False,
    ignore=["input", "select", "textarea"],
)

ui.run(
    title="Gesture Dataset Labeler",
    reload=False,
)
