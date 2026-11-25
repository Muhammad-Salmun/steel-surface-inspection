from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path("runs/detect/steel_detector4/weights/best.pt")

# Edit these values for your inspection setup.
EXPECTED_SURFACE_NUMBER = "9"
MM_PER_PIXEL = 0.05
MIN_WIDTH_MM = 35.0
MAX_WIDTH_MM = 45.0
MIN_HEIGHT_MM = 40.0
MAX_HEIGHT_MM = 45.0
DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 720

BOX_COLOR_OK = (40, 210, 90)
BOX_COLOR_ERROR = (40, 40, 230)
BOX_COLOR_WARNING = (0, 180, 255)
PANEL_BG = (22, 28, 36)
TEXT_COLOR = (245, 245, 245)
MUTED_TEXT = (180, 190, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inspection inference and flag wrong roughness or dimensions."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to model weights. Default: {DEFAULT_WEIGHTS}",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index to use. Default: 0",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path for single-image testing instead of webcam.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cpu', '0', or 'cuda:0'. Default: auto",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save annotated image or webcam video.",
    )
    return parser.parse_args()


def clean_display_text(value: str) -> str:
    return value.encode("ascii", errors="replace").decode("ascii")


def surface_number_from_label(label: str) -> str | None:
    match = re.search(r"(\d+)$", label)
    return match.group(1) if match else None


def draw_text(
    image,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        clean_display_text(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_box_label(
    image,
    x1: int,
    y1: int,
    lines: list[str],
    color: tuple[int, int, int],
) -> None:
    font_scale = 0.55
    thickness = 1
    lines = [clean_display_text(line) for line in lines if line]
    if not lines:
        return

    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        for line in lines
    ]
    label_width = max(size[0][0] for size in sizes)
    line_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    padding_x = 6
    padding_y = 5
    line_gap = 4
    label_height = len(lines) * line_height + (len(lines) - 1) * line_gap + baseline

    image_width = image.shape[1]
    left = min(max(x1, 0), max(image_width - label_width - padding_x * 2, 0))
    top = max(y1 - label_height - padding_y * 2, 0)
    cv2.rectangle(
        image,
        (left, top),
        (left + label_width + padding_x * 2, top + label_height + padding_y * 2),
        color,
        -1,
    )

    text_y = top + padding_y + line_height
    for line in lines:
        draw_text(image, line, (left + padding_x, text_y), font_scale, (20, 20, 20), thickness)
        text_y += line_height + line_gap


def dimensions_are_valid(width_mm: float, height_mm: float) -> bool:
    return (
        MIN_WIDTH_MM <= width_mm <= MAX_WIDTH_MM
        and MIN_HEIGHT_MM <= height_mm <= MAX_HEIGHT_MM
    )


def inspect_detection(class_label: str, width_px: int, height_px: int) -> tuple[list[str], bool]:
    surface_number = surface_number_from_label(class_label) or class_label
    width_mm = width_px * MM_PER_PIXEL
    height_mm = height_px * MM_PER_PIXEL

    errors: list[str] = []
    if surface_number != EXPECTED_SURFACE_NUMBER:
        errors.append("ROUGHNESS ERROR")
    if not dimensions_are_valid(width_mm, height_mm):
        errors.append("DIMENSION ERROR")

    status_ok = not errors
    status = "OK" if status_ok else " + ".join(errors)
    lines = [
        f"{status}",
        f"Roughness: {surface_number} | Expected: {EXPECTED_SURFACE_NUMBER}",
        f"Size: {width_mm:.1f} x {height_mm:.1f} mm",
    ]
    return lines, status_ok


def draw_status_panel(
    image,
    detections_count: int,
    errors_count: int,
    fps: float | None = None,
) -> None:
    panel_width = 420
    panel_height = 145
    x = 12
    y = 12

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)

    if detections_count == 0:
        status_text = "NO DETECTION"
        status_color = BOX_COLOR_WARNING
    elif errors_count == 0:
        status_text = "INSPECTION OK"
        status_color = BOX_COLOR_OK
    else:
        status_text = "INSPECTION ERROR"
        status_color = BOX_COLOR_ERROR

    cv2.rectangle(image, (x, y), (x + panel_width, y + 5), status_color, -1)
    draw_text(image, status_text, (x + 16, y + 36), 0.82, status_color, 2)
    draw_text(image, f"Expected roughness: {EXPECTED_SURFACE_NUMBER}", (x + 16, y + 68), 0.55)
    draw_text(
        image,
        f"Expected dimension: {MIN_WIDTH_MM:.1f}-{MAX_WIDTH_MM:.1f} x "
        f"{MIN_HEIGHT_MM:.1f}-{MAX_HEIGHT_MM:.1f} mm",
        (x + 16, y + 96),
        0.55,
        MUTED_TEXT,
    )
    draw_text(image, f"Scale: {MM_PER_PIXEL:.4f} mm/px", (x + 16, y + 124), 0.55, MUTED_TEXT)
    if fps is not None:
        draw_text(image, f"FPS: {fps:.1f}", (x + panel_width - 95, y + 124), 0.5, MUTED_TEXT)


def draw_inspection(image, result, model_names: dict[int, str]) -> tuple[int, int]:
    detections_count = 0
    errors_count = 0

    for box in result.boxes:
        detections_count += 1
        class_id = int(box.cls[0])
        class_label = model_names.get(class_id, str(class_id))
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        width_px = x2 - x1
        height_px = y2 - y1

        label_lines, status_ok = inspect_detection(class_label, width_px, height_px)
        color = BOX_COLOR_OK if status_ok else BOX_COLOR_ERROR
        if not status_ok:
            errors_count += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_box_label(image, x1, y1, label_lines, color)

    return detections_count, errors_count


def resize_for_display(image):
    if DISPLAY_WIDTH <= 0 and DISPLAY_HEIGHT <= 0:
        return image

    height, width = image.shape[:2]
    width_scale = DISPLAY_WIDTH / width if DISPLAY_WIDTH > 0 else 1.0
    height_scale = DISPLAY_HEIGHT / height if DISPLAY_HEIGHT > 0 else 1.0
    scale = min(width_scale, height_scale, 1.0)
    if scale >= 1.0:
        return image

    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def run_single_image(args: argparse.Namespace, model: YOLO) -> None:
    image_path = args.image.expanduser()
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = model.predict(
        source=frame,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )[0]
    annotated_frame = frame.copy()
    detections_count, errors_count = draw_inspection(annotated_frame, result, model.names)
    draw_status_panel(annotated_frame, detections_count, errors_count)

    if args.save is not None:
        cv2.imwrite(str(args.save), annotated_frame)
        print(f"Saved annotated image to: {args.save}")

    display_frame = resize_for_display(annotated_frame)
    cv2.imshow("Quality Check - press any key to close", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(args: argparse.Namespace, model: YOLO) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {args.camera}")

    writer = None
    if args.save is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, fps, (width, height))

    window_name = "Quality Check - press q or Esc to quit"
    previous_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            result = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            annotated_frame = frame.copy()
            detections_count, errors_count = draw_inspection(annotated_frame, result, model.names)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            draw_status_panel(annotated_frame, detections_count, errors_count, fps)

            if writer is not None:
                writer.write(annotated_frame)

            display_frame = resize_for_display(annotated_frame)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    weights_path = args.weights.expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    if args.image is not None:
        run_single_image(args, model)
    else:
        run_webcam(args, model)


if __name__ == "__main__":
    main()
