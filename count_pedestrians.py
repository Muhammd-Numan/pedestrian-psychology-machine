import os
import cv2
import time
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import Font
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
VIDEO_PATH = "JAAD_clips/video_0017.mp4"
JAAD_XML_PATH = "JAAD_annotations/annotations/video_0017.xml"             # set this to your JAAD .xml to get a GT target count
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolov8m.pt"        # try yolov8l.pt if GPU allows -> fewer missed peds
PERSON_CLASS_ID = 0

# --- detection ---
CONF_THRESHOLD = 0.25            # LOW on purpose: catches small/far pedestrians.
                                 # Ghost tracks are prevented by NEW_TRACK_THRESH
                                 # below, not by this value.
IMG_SIZE = 1280                  # try 1536 if undercounting far pedestrians

# --- tracker (anti-fragmentation) ---
NEW_TRACK_THRESH = 0.70          # a NEW ID is only created from a detection this
                                 # confident. Kills flicker IDs at the source.
TRACK_BUFFER = 300               # frames a lost track stays alive for re-claiming
USE_REID = True                  # appearance re-ID (ultralytics >= 8.3.114)

# --- post-processing: merge fragmented tracks ---
MERGE_GAP_FRAMES = 90            # if track B starts <= this many frames after
                                 # track A ended...
MERGE_IOU = 0.1                 # ...and B's first box overlaps A's last box by
                                 # at least this IoU, A and B are the same person.

# --- final filter ---
MIN_TRACK_SECONDS = 1.0          # an ID must exist >= this long (in seconds) to
                                 # count. FPS-relative, so it works for any video.
# ----------------------------


def parse_jaad_xml(xml_path):
    """Return (gt_by_frame, gt_unique_count, group_track_count).

    gt_unique_count = number of distinct 'pedestrian'/'ped' tracks.
    group_track_count = number of 'people' (crowd/group) tracks. The model
    detects individuals inside these groups, which inflates your count vs GT.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_by_frame = defaultdict(list)
    unique_tracks = 0
    group_tracks = 0

    for track in root.findall("track"):
        label = track.get("label")
        if label == "people":
            group_tracks += 1
            continue
        if label not in ("pedestrian", "ped"):
            continue
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
    return gt_by_frame, unique_tracks, group_tracks


def build_tracker_config():
    """Copy the installed default botsort.yaml and override the keys that
    reduce ID fragmentation."""
    import ultralytics
    default_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    cfg["track_buffer"] = TRACK_BUFFER
    cfg["new_track_thresh"] = NEW_TRACK_THRESH
    cfg["gmc_method"] = "sparseOptFlow"   # camera-motion compensation (ego-vehicle!)
    if USE_REID:
        cfg["with_reid"] = True
        cfg["model"] = "yolov8n-cls.pt"   # real appearance model instead of "auto"
        cfg["proximity_thresh"] = 0.2     # was 0.5 — allow ReID even if he moved while hidden
        cfg["appearance_thresh"] = 0.25   # how similar he must look to reclaim his old ID

    custom_path = OUTPUT_DIR / "custom_botsort.yaml"
    with open(custom_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(custom_path)


def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def merge_fragmented_tracks(frame_detections):
    """Post-processing pass: if a track ends and another starts shortly after
    at roughly the same place, they are the same person. Returns remapped
    detections and the number of merges performed."""
    info = {}
    for fidx, dets in enumerate(frame_detections):
        for (x1, y1, x2, y2, tid) in dets:
            box = (x1, y1, x2, y2)
            if tid not in info:
                info[tid] = {"first": fidx, "last": fidx,
                             "first_box": box, "last_box": box}
            else:
                info[tid]["last"] = fidx
                info[tid]["last_box"] = box

    parent = {tid: tid for tid in info}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    merges = 0
    ordered = sorted(info.keys(), key=lambda t: info[t]["first"])
    for tid_b in ordered:
        fb = info[tid_b]["first"]
        best, best_iou = None, MERGE_IOU
        for tid_a in ordered:
            if tid_a == tid_b:
                continue
            root_a = find(tid_a)
            if root_a == find(tid_b):
                continue
            la = info[root_a]["last"]
            if la < fb and (fb - la) <= MERGE_GAP_FRAMES:
                iou = compute_iou(info[root_a]["last_box"], info[tid_b]["first_box"])
                if iou >= best_iou:
                    best, best_iou = root_a, iou
        if best is not None:
            parent[find(tid_b)] = best
            merges += 1
            if info[tid_b]["last"] > info[best]["last"]:
                info[best]["last"] = info[tid_b]["last"]
                info[best]["last_box"] = info[tid_b]["last_box"]

    remapped = [[(x1, y1, x2, y2, find(tid)) for (x1, y1, x2, y2, tid) in dets]
                for dets in frame_detections]
    return remapped, merges


def main():
    model = YOLO(MODEL_NAME)
    tracker_cfg = build_tracker_config()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    gt_unique_count = None
    gt_group_tracks = 0
    if JAAD_XML_PATH and Path(JAAD_XML_PATH).exists():
        print(f"Loading JAAD annotations: {JAAD_XML_PATH}")
        gt_by_frame, gt_unique_count, gt_group_tracks = parse_jaad_xml(JAAD_XML_PATH)
        total_gt_boxes = sum(len(v) for v in gt_by_frame.values())
        print(f"  GT: {gt_unique_count} unique pedestrians across "
              f"{len(gt_by_frame)} frames ({total_gt_boxes} boxes)")
        if gt_group_tracks:
            print(f"  NOTE: {gt_group_tracks} 'people' (group) tracks exist — the "
                  f"model will count individuals inside these groups, so expect "
                  f"a higher model count on this clip.")
    else:
        print("No JAAD annotations provided — running detection only.")
        gt_by_frame = {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_track_frames = max(10, int(round(fps * MIN_TRACK_SECONDS)))
    print(f"Video: {width}x{height} | {fps:.1f} FPS | {total_frames} frames")
    print(f"Min track length to count: {min_track_frames} frames "
          f"({MIN_TRACK_SECONDS:.1f}s)")

    video_name = Path(VIDEO_PATH).stem

    # ============================================================
    # PASS 1 — detection + tracking
    # ============================================================
    frame_detections = []
    frame_fps = []

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
                imgsz=IMG_SIZE,
                verbose=False,
            )

            dets = []
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                ids_this_frame = boxes.id.cpu().numpy().astype(int).tolist()
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

    # ---------- post-processing ----------
    raw_unique = len({tid for dets in frame_detections for (*_, tid) in dets})

    frame_detections, n_merges = merge_fragmented_tracks(frame_detections)
    merged_unique = len({tid for dets in frame_detections for (*_, tid) in dets})

    id_frame_counts = defaultdict(int)
    for dets in frame_detections:
        for (*_, tid) in dets:
            id_frame_counts[tid] += 1

    survivors = {tid for tid, c in id_frame_counts.items() if c >= min_track_frames}
    filtered_unique = len(survivors)

    # gapless 1..N labels for surviving tracks, in order of first appearance
    display_ids = {}
    for dets in frame_detections:
        for (*_, tid) in dets:
            if tid in survivors and tid not in display_ids:
                display_ids[tid] = len(display_ids) + 1

    # ============================================================
    # PASS 2 — render comparison video (no model, fast)
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

            for (x1, y1, x2, y2) in gt_by_frame.get(frame_idx, []):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "JAAD", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for (x1, y1, x2, y2, tid) in frame_detections[frame_idx]:
                if tid not in survivors:
                    continue
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
    header = ["video_id", "unique_people", "after_merge", "raw_track_ids",
              "merges", "gt_unique_people", "gt_group_tracks",
              "avg_processing_fps"]
    ws.append(header)
    ws.append([video_name, filtered_unique, merged_unique, raw_unique,
               n_merges,
               gt_unique_count if gt_unique_count is not None else "N/A",
               gt_group_tracks, round(avg_fps, 2)])

    bold = Font(bold=True)
    for col_idx in range(1, len(header) + 1):
        ws.cell(row=1, column=col_idx).font = bold
    for letter, w in zip("ABCDEFGH", (30, 14, 12, 14, 10, 16, 16, 18)):
        ws.column_dimensions[letter].width = w
    wb.save(summary_path)

    # ---------- report ----------
    print("\n--- SUMMARY ---")
    print(f"Raw tracker IDs:                    {raw_unique}")
    print(f"After fragmentation merge (-{n_merges}):     {merged_unique}")
    print(f"Unique people (>= {min_track_frames} frames):       {filtered_unique}")
    if gt_unique_count is not None:
        print(f"JAAD ground-truth unique:           {gt_unique_count}"
              + (f" (+{gt_group_tracks} group tracks)" if gt_group_tracks else ""))
        print(f"Difference:                         {filtered_unique - gt_unique_count:+d}")
    print(f"Average processing FPS:             {avg_fps:.2f}")
    print(f"\nOutputs:")
    print(f"  Comparison video: {out_video_path}")
    print(f"  Excel summary:    {summary_path}")


if __name__ == "__main__":
    main()