import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIG ----------
VIDEO_PATH = "test_pedestrians.mp4"
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
    frame_records = []
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
            num_people = len(boxes)
            frame_records.append({"frame": frame_idx, "person_count": num_people})

            if boxes.id is not None:
                unique_ids.update(boxes.id.cpu().numpy().astype(int).tolist())

            annotated = frame.copy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"ID:{track_id} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            cv2.putText(
                annotated,
                f"People in frame: {num_people}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"Unique so far: {len(unique_ids)}",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"Frame: {frame_idx}",
                (15, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            writer.write(annotated)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    df = pd.DataFrame(frame_records)
    csv_path = OUTPUT_DIR / f"{video_name}_counts.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- SUMMARY ---")
    print(f"Total frames processed:        {len(df)}")
    print(f"Average people per frame:      {df['person_count'].mean():.2f}")
    print(f"Max people in a frame:         {df['person_count'].max()}")
    print(f"Min people in a frame:         {df['person_count'].min()}")
    print(f"Frames with 0 people:          {(df['person_count'] == 0).sum()}")
    print(f"Unique pedestrians (tracked):  {len(unique_ids)}")
    print(f"\nOutputs saved:")
    print(f"  Annotated video: {out_video_path}")
    print(f"  Per-frame CSV:   {csv_path}")


if __name__ == "__main__":
    main()