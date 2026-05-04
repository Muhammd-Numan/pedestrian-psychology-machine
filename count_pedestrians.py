import cv2
from openpyxl import Workbook
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
VIDEO_PATH = "video_0003.mp4"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.4
PERSON_CLASS_ID = 0
# ----------------------------


def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing: {VIDEO_PATH}")
    print(f"  {width}x{height} | {fps:.1f} FPS | {total_frames} frames")

    video_name = Path(VIDEO_PATH).stem
    out_video_path = OUTPUT_DIR / f"{video_name}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    unique_ids = set()
    frame_idx = 0

    with tqdm(total=total_frames, desc="Detecting + Tracking") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            boxes = results[0].boxes

            if boxes.id is not None:
                unique_ids.update(boxes.id.cpu().numpy().astype(int).tolist())

            annotated = frame.copy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"ID:{track_id}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            cv2.putText(
                annotated,
                f"Unique people so far: {len(unique_ids)}",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            writer.write(annotated)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    # Save final summary as Excel
    summary_path = OUTPUT_DIR / f"{video_name}_summary.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["video_id", "number_of_unique_people"])
    ws.append([video_name, len(unique_ids)])

    
    from openpyxl.styles import Font
    bold_font = Font(bold=True)
    ws["A1"].font = bold_font
    ws["B1"].font = bold_font

    
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 25

    wb.save(summary_path)

    print("\n--- SUMMARY ---")
    print(f"Total unique pedestrians in video: {len(unique_ids)}")
    print(f"\nOutputs saved:")
    print(f"  Annotated video: {out_video_path}")
    print(f"  Summary Excel:   {summary_path}")


if __name__ == "__main__":
    main()