import os
import cv2
import time
import yaml
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import Font
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
VIDEO_PATH = "Sample_1.mp4"
JAAD_XML_PATH = None             # set this to your JAAD .xml to get a GT target count
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolov8m.pt"
CONF_THRESHOLD = 0.4             # detection confidence
PERSON_CLASS_ID = 0

# --- knobs that control the unique-people count ---
MIN_TRACK_FRAMES = 35            # an ID must be seen at least this many frames to count
                                 #   as a real person (kills flicker / false positives).
                                 #   Tune this to make the count match GT.
TRACK_BUFFER = 150               # how long (frames) a lost track is kept alive so a
                                 #   re-appearing person can reclaim its old ID.
USE_REID = True                  # appearance-based re-identification (needs ultralytics
                                 #   >= 8.3.114). Set False if your version is older.
# ----------------------------


def parse_jaad_xml(xml_path):
    """Return (gt_by_frame, gt_unique_count).

    gt_by_frame: {frame_idx: [(x1, y1, x2, y2), ...]} for drawing.
    gt_unique_count: number of distinct pedestrian *tracks* = real unique people.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_by_frame = defaultdict(list)
    unique_tracks = 0

    for track in root.findall("track"):
        label = track.get("label")
        if label not in ("pedestrian", "ped"):
            continue
        # one <track> element == one unique annotated person
        unique_tracks += 1
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame_idx = int(box.get("frame"))
            x1 = int(float(box.get("xtl")))
            y1 = int(float(box.get("ytl")))
            x2 = int(float(box.get("xbr")))
            y2 = int(float(box.get("ybr")))
            gt_by_frame[frame_idx].append((x1, y1, x2, y2))
    return gt_by_frame, unique_tracks


def build_tracker_config():
    """Copy the installed default botsort.yaml and override the keys that reduce
    ID fragmentation. Building from the installed file guarantees the correct
    defaults for whatever ultralytics version is present."""
    import ultralytics
    default_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    cfg["track_buffer"] = TRACK_BUFFER
    if USE_REID:
        cfg["with_reid"] = True
        cfg["model"] = "auto"          # use the detector's own features, ~no extra latency

    custom_path = OUTPUT_DIR / "custom_botsort.yaml"
    with open(custom_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(custom_path)


def main():
    model = YOLO(MODEL_NAME)
    tracker_cfg = build_tracker_config()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    gt_unique_count = None
    if JAAD_XML_PATH and Path(JAAD_XML_PATH).exists():
        print(f"Loading JAAD annotations: {JAAD_XML_PATH}")
        gt_by_frame, gt_unique_count = parse_jaad_xml(JAAD_XML_PATH)
        total_gt_boxes = sum(len(v) for v in gt_by_frame.values())
        print(f"  GT: {gt_unique_count} unique pedestrians across "
              f"{len(gt_by_frame)} frames ({total_gt_boxes} boxes)")
    else:
        print("No JAAD annotations provided — running detection only.")
        gt_by_frame = {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} | {fps:.1f} FPS | {total_frames} frames")

    video_name = Path(VIDEO_PATH).stem

    # ============================================================
    # PASS 1 — detection + tracking only (the expensive part).
    # We store every frame's detections so we can re-draw later
    # without running the model a second time.
    # ============================================================
    id_frame_counts = defaultdict(int)   # {raw_track_id: number of frames it appeared in}
    frame_detections = []                # frame_detections[i] = [(x1, y1, x2, y2, raw_tid), ...]
    frame_fps = []                       # detection FPS measured per frame

    with tqdm(total=total_frames, desc="Detecting") as pbar:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                persist=True,
                tracker=tracker_cfg,
                imgsz=1280,
                verbose=False,
            )

            dets = []
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                ids_this_frame = boxes.id.cpu().numpy().astype(int).tolist()
                for tid in ids_this_frame:
                    id_frame_counts[tid] += 1
                for box, tid in zip(boxes, ids_this_frame):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    dets.append((x1, y1, x2, y2, tid))

            frame_detections.append(dets)

            elapsed = time.time() - start_time
            frame_fps.append(1.0 / elapsed if elapsed > 0 else 0)
            pbar.update(1)

    cap.release()
    total_frames_read = len(frame_detections)

    avg_fps = sum(frame_fps) / len(frame_fps) if frame_fps else 0

    # ---------- counting + build the gapless 1..N label map ----------
    raw_unique = len(id_frame_counts)
    survivors = {tid for tid, c in id_frame_counts.items() if c >= MIN_TRACK_FRAMES}
    filtered_unique = len(survivors)

    # Assign clean sequential labels (1, 2, 3, ... N) to the surviving tracks
    # only, in the order they first appear. Ghost/flicker tracks are skipped, so
    # the numbers are gapless and the highest label == filtered_unique.
    display_ids = {}
    for dets in frame_detections:
        for (x1, y1, x2, y2, tid) in dets:
            if tid in survivors and tid not in display_ids:
                display_ids[tid] = len(display_ids) + 1

    # ============================================================
    # PASS 2 — re-read the frames and draw. No model here, so this
    # is just I/O and is fast even on CPU.
    # ============================================================
    out_video_path = OUTPUT_DIR / f"{video_name}_comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    cap = cv2.VideoCapture(VIDEO_PATH)
    with tqdm(total=total_frames_read, desc="Rendering") as pbar:
        for frame_idx in range(total_frames_read):
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()

            # RED boxes = JAAD ground truth
            for (x1, y1, x2, y2) in gt_by_frame.get(frame_idx, []):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "JAAD", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # GREEN boxes = my model — only the counted people, labeled 1..N
            for (x1, y1, x2, y2, tid) in frame_detections[frame_idx]:
                if tid not in survivors:
                    continue                      # skip filtered-out flicker tracks
                label = str(display_ids[tid])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(annotated, "GREEN = My model   RED = JAAD ground truth",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Processing FPS: {frame_fps[frame_idx]:.1f}",
                        (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            writer.write(annotated)
            pbar.update(1)

    cap.release()
    writer.release()

    # ---------- Excel summary ----------
    summary_path = OUTPUT_DIR / f"{video_name}_summary.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    header = ["video_id", "unique_people", "raw_track_ids",
              "gt_unique_people", "avg_processing_fps"]
    ws.append(header)
    ws.append([video_name, filtered_unique, raw_unique,
               gt_unique_count if gt_unique_count is not None else "N/A",
               round(avg_fps, 2)])

    bold = Font(bold=True)
    for col_idx in range(1, len(header) + 1):
        ws.cell(row=1, column=col_idx).font = bold
    for letter, w in zip("ABCDE", (30, 16, 16, 18, 22)):
        ws.column_dimensions[letter].width = w
    wb.save(summary_path)

    # ---------- report ----------
    print("\n--- SUMMARY ---")
    print(f"Raw tracker IDs (overcounts):  {raw_unique}")
    print(f"Unique people (>= {MIN_TRACK_FRAMES} frames): {filtered_unique}")
    if gt_unique_count is not None:
        print(f"JAAD ground-truth unique:      {gt_unique_count}")
        print(f"Difference:                    {filtered_unique - gt_unique_count:+d}")
    print(f"Average processing FPS:        {avg_fps:.2f}")
    print(f"\nOutputs:")
    print(f"  Comparison video: {out_video_path}")
    print(f"  Excel summary:    {summary_path}")


if __name__ == "__main__":
    main()