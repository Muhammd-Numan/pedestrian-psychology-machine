import cv2
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import Font
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
VIDEO_PATH = "video_0001.mp4"
JAAD_XML_PATH = "JAAD_annotations/annotations/video_0001.xml"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.4
PERSON_CLASS_ID = 0
# ----------------------------


def parse_jaad_xml(xml_path):
    """Return dict: {frame_idx: [(x1, y1, x2, y2), ...]}"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_by_frame = defaultdict(list)

    for track in root.findall("track"):
        label = track.get("label")
        if label not in ("pedestrian", "ped"):
            continue
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame_idx = int(box.get("frame"))
            x1 = int(float(box.get("xtl")))
            y1 = int(float(box.get("ytl")))
            x2 = int(float(box.get("xbr")))
            y2 = int(float(box.get("ybr")))
            gt_by_frame[frame_idx].append((x1, y1, x2, y2))
    return gt_by_frame


def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    print(f"Loading JAAD annotations: {JAAD_XML_PATH}")
    gt_by_frame = parse_jaad_xml(JAAD_XML_PATH)
    total_gt_boxes = sum(len(v) for v in gt_by_frame.values())
    print(f"  Loaded GT for {len(gt_by_frame)} frames, {total_gt_boxes} total boxes")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} | {fps:.1f} FPS | {total_frames} frames")

    video_name = Path(VIDEO_PATH).stem
    out_video_path = OUTPUT_DIR / f"{video_name}_comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    unique_ids = set()
    frame_idx = 0
    processing_fps_history = []  # for averaging at the end

    with tqdm(total=total_frames, desc="Processing") as pbar:
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
                tracker="bytetrack.yaml",
                verbose=False,
            )

            annotated = frame.copy()

            # RED boxes = JAAD ground truth (from XML)
            for (x1, y1, x2, y2) in gt_by_frame.get(frame_idx, []):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, "JAAD", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # GREEN boxes = my YOLO detections
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                if boxes.id is not None:
                    unique_ids.update(boxes.id.cpu().numpy().astype(int).tolist())

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    track_id = int(box.id[0]) if box.id is not None else -1
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"YOLO ID:{track_id}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Calculate processing FPS for this frame
            elapsed = time.time() - start_time
            processing_fps = 1.0 / elapsed if elapsed > 0 else 0
            processing_fps_history.append(processing_fps)

            # Legend + live count + processing FPS
            cv2.putText(annotated, "GREEN = My model   RED = JAAD ground truth",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated, f"Unique people so far: {len(unique_ids)}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(annotated, f"Processing FPS: {processing_fps:.1f}",
                        (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            writer.write(annotated)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    avg_fps = sum(processing_fps_history) / len(processing_fps_history) if processing_fps_history else 0

    # ---------- Excel summary ----------
    summary_path = OUTPUT_DIR / f"{video_name}_summary.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["video_id", "number_of_unique_people", "avg_processing_fps"])
    ws.append([video_name, len(unique_ids), round(avg_fps, 2)])

    bold = Font(bold=True)
    for col in ("A1", "B1", "C1"):
        ws[col].font = bold
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 25
    ws.column_dimensions["C"].width = 22

    wb.save(summary_path)

    print("\n--- SUMMARY ---")
    print(f"Total unique pedestrians: {len(unique_ids)}")
    print(f"Average processing FPS:   {avg_fps:.2f}")
    print(f"\nOutputs:")
    print(f"  Comparison video: {out_video_path}")
    print(f"  Excel summary:    {summary_path}")


if __name__ == "__main__":
    main()