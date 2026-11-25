from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from xml.etree import ElementTree
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path("runs/detect/steel_detector4/weights/best.pt")
DEFAULT_DATA_MAP = Path("data output map")
BOX_COLOR = (0, 180, 255)
PANEL_BG = (22, 28, 36)
PANEL_ACCENT = (0, 180, 255)
TEXT_COLOR = (245, 245, 245)
MUTED_TEXT = (180, 190, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run webcam inference with the trained steel surface detector."
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
        help="Optional path to save annotated webcam video, for example output.mp4",
    )
    parser.add_argument(
        "--data-map",
        type=Path,
        default=DEFAULT_DATA_MAP,
        help=f"Folder containing lab data spreadsheets. Default: {DEFAULT_DATA_MAP}",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1100,
        help="Maximum display window width. Use 0 for original size. Default: 1100",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=720,
        help="Maximum display window height. Use 0 for original size. Default: 720",
    )
    parser.add_argument(
        "--show-dimensions-mm",
        action="store_true",
        help="Show detected box width and height in millimeters.",
    )
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=None,
        help="Camera calibration scale for box dimensions, in millimeters per pixel.",
    )
    return parser.parse_args()


def surface_number_from_label(label: str) -> str | None:
    match = re.search(r"(\d+)$", label)
    return match.group(1) if match else None


def find_lab_file(data_map_dir: Path, surface_number: str) -> Path | None:
    for suffix in (".xlsx", ".xls", ".csv"):
        candidate = data_map_dir / f"{surface_number}{suffix}"
        if candidate.exists():
            return candidate
    return None


def convert_spreadsheet_to_csv(source: Path, output_dir: Path) -> Path:
    converter = shutil.which("libreoffice") or shutil.which("soffice")
    if converter is None:
        raise RuntimeError(
            "LibreOffice was not found. Install libreoffice, or convert the lab "
            "spreadsheets in 'data output map' to CSV files."
        )

    subprocess.run(
        [
            converter,
            "--headless",
            "--convert-to",
            "csv",
            "--outdir",
            str(output_dir),
            str(source),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    converted = output_dir / f"{source.stem}.csv"
    if not converted.exists():
        raise RuntimeError(f"Could not convert lab data file to CSV: {source}")
    return converted


def clean_cell(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").strip().split())


def clean_display_text(value: str) -> str:
    replacements = {
        "µ": "u",
        "λ": "lambda",
        "–": "-",
        "—": "-",
        "−": "-",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value.encode("ascii", errors="replace").decode("ascii")


def read_report_csv(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as csv_file:
        rows = [[clean_cell(cell) for cell in row] for row in csv.reader(csv_file)]

    return read_report_rows(rows)


def spreadsheet_column_index(cell_reference: str) -> int:
    letters = "".join(char for char in cell_reference if char.isalpha())
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter.upper()) - ord("A") + 1
    return max(index - 1, 0)


def read_xlsx_rows(xlsx_path: Path) -> list[list[str]]:
    namespace = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    with zipfile.ZipFile(xlsx_path) as workbook_zip:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook_zip.namelist():
            shared_root = ElementTree.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("main:si", namespace):
                parts = [node.text or "" for node in item.findall(".//main:t", namespace)]
                shared_strings.append(clean_cell("".join(parts)))

        workbook_root = ElementTree.fromstring(workbook_zip.read("xl/workbook.xml"))
        first_sheet = workbook_root.find("main:sheets/main:sheet", namespace)
        if first_sheet is None:
            return []

        relation_id = first_sheet.attrib.get(f"{{{namespace['rel']}}}id")
        relationships_root = ElementTree.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))
        sheet_target = None
        for relation in relationships_root.findall("pkgrel:Relationship", namespace):
            if relation.attrib.get("Id") == relation_id:
                sheet_target = relation.attrib.get("Target")
                break

        if not sheet_target:
            return []

        sheet_path = f"xl/{sheet_target.lstrip('/')}"
        sheet_root = ElementTree.fromstring(workbook_zip.read(sheet_path))
        rows: list[list[str]] = []
        for row in sheet_root.findall(".//main:sheetData/main:row", namespace):
            values: list[str] = []
            for cell in row.findall("main:c", namespace):
                column_index = spreadsheet_column_index(cell.attrib.get("r", "A1"))
                while len(values) <= column_index:
                    values.append("")

                cell_type = cell.attrib.get("t")
                if cell_type == "s":
                    raw_value = cell.findtext("main:v", default="", namespaces=namespace)
                    value = shared_strings[int(raw_value)] if raw_value else ""
                elif cell_type == "inlineStr":
                    parts = [node.text or "" for node in cell.findall(".//main:t", namespace)]
                    value = "".join(parts)
                else:
                    value = cell.findtext("main:v", default="", namespaces=namespace)

                values[column_index] = clean_cell(value)
            rows.append(values)

    return rows


def read_report_rows(rows: list[list[str]]) -> dict[str, str]:

    data: dict[str, str] = {}
    for row in rows:
        cells = [cell for cell in row if cell]
        if len(cells) == 1 and re.search(r"\d{1,2}\s*-\s*[A-Za-z]{3}", cells[0]):
            data["Date"] = cells[0]

        for index, cell in enumerate(row[:-1]):
            if cell in {"Standard", "Profile", "Cut-Off", "Filter", "Ra", "Rq", "Rz", "N"}:
                value = clean_cell(row[index + 1])
                if value:
                    data[cell] = value

        for index, cell in enumerate(row[:-2]):
            if cell == "λs":
                value = clean_cell(row[index + 1])
                if value:
                    data["Lambda s"] = value

    return data


def load_lab_data(data_map_dir: Path, model_names: dict[int, str]) -> dict[int, dict[str, str]]:
    if not data_map_dir.exists():
        print(f"Lab data folder not found: {data_map_dir}")
        return {}

    lab_data: dict[int, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="steel_lab_data_") as temp_dir:
        temp_path = Path(temp_dir)
        for class_id, label in model_names.items():
            surface_number = surface_number_from_label(label)
            if surface_number is None:
                continue

            lab_file = find_lab_file(data_map_dir, surface_number)
            if lab_file is None:
                continue

            try:
                if lab_file.suffix == ".csv":
                    report_data = read_report_csv(lab_file)
                elif lab_file.suffix == ".xlsx":
                    report_data = read_report_rows(read_xlsx_rows(lab_file))
                else:
                    csv_path = convert_spreadsheet_to_csv(lab_file, temp_path)
                    report_data = read_report_csv(csv_path)

                lab_data[class_id] = report_data
                lab_data[class_id]["Surface"] = surface_number
            except Exception as exc:
                print(f"Could not read lab data for surface {surface_number}: {exc}")

    return lab_data


def draw_text(
    image,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 1,
) -> None:
    text = clean_display_text(text)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_box_label(image, x1: int, y1: int, text: str | list[str]) -> None:
    font_scale = 0.55
    thickness = 1
    lines = [text] if isinstance(text, str) else text
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
        BOX_COLOR,
        -1,
    )
    text_y = top + padding_y + line_height
    for line in lines:
        draw_text(image, line, (left + padding_x, text_y), font_scale, (20, 20, 20), thickness)
        text_y += line_height + line_gap


def draw_lab_panel(image, lab_info: dict[str, str] | None, confidence: float, fps: float) -> None:
    height, width = image.shape[:2]
    panel_width = min(380, width - 24)
    panel_height = 300
    x = width - panel_width - 12
    y = 12

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)
    cv2.rectangle(image, (x, y), (x + panel_width, y + 5), PANEL_ACCENT, -1)

    if not lab_info:
        draw_text(image, "Steel Surface Detector", (x + 16, y + 34), 0.7)
        draw_text(image, "No surface detected", (x + 16, y + 70), 0.62, MUTED_TEXT)
        draw_text(image, f"FPS: {fps:.1f}", (x + 16, y + panel_height - 18), 0.55, MUTED_TEXT)
        return

    surface = lab_info.get("Surface", "-")
    draw_text(image, f"Surface {surface}", (x + 16, y + 36), 0.82, TEXT_COLOR, 2)
    draw_text(image, f"Confidence: {confidence:.2f}", (x + 16, y + 66), 0.55, MUTED_TEXT)

    rows = [
        ("Ra", lab_info.get("Ra", "-")),
        ("Rq", lab_info.get("Rq", "-")),
        ("Rz", lab_info.get("Rz", "-")),
        ("Cut-Off", lab_info.get("Cut-Off", "-")),
        ("Filter", lab_info.get("Filter", "-")),
        ("Standard", lab_info.get("Standard", "-")),
    ]
    row_y = y + 100
    for label, value in rows:
        draw_text(image, label, (x + 16, row_y), 0.58, MUTED_TEXT)
        draw_text(image, value, (x + 135, row_y), 0.58, TEXT_COLOR)
        row_y += 26

    footer = lab_info.get("Date", "")
    if footer:
        draw_text(image, footer, (x + 16, y + panel_height - 18), 0.5, MUTED_TEXT)
    draw_text(image, f"FPS: {fps:.1f}", (x + panel_width - 96, y + panel_height - 18), 0.5, MUTED_TEXT)


def format_box_dimensions_mm(width_px: int, height_px: int, mm_per_pixel: float) -> str:
    width_mm = width_px * mm_per_pixel
    height_mm = height_px * mm_per_pixel
    return f"Box {width_mm:.1f} x {height_mm:.1f} mm"


def draw_detections(
    image,
    result,
    lab_data: dict[int, dict[str, str]],
    mm_per_pixel: float | None = None,
) -> tuple[dict[str, str] | None, float]:
    active_lab_info = None
    active_confidence = 0.0

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        lab_info = lab_data.get(class_id)
        surface = lab_info.get("Surface", str(class_id)) if lab_info else str(class_id)
        ra = lab_info.get("Ra", "-") if lab_info else "-"
        rz = lab_info.get("Rz", "-") if lab_info else "-"
        label = [f"Surface {surface} | Ra {ra} | Rz {rz}"]
        if mm_per_pixel is not None:
            label.append(format_box_dimensions_mm(x2 - x1, y2 - y1, mm_per_pixel))

        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        draw_box_label(image, x1, y1, label)

        if confidence > active_confidence:
            active_confidence = confidence
            active_lab_info = lab_info or {"Surface": surface}

    return active_lab_info, active_confidence


def resize_for_display(image, max_width: int, max_height: int):
    if max_width <= 0 and max_height <= 0:
        return image

    height, width = image.shape[:2]
    width_scale = max_width / width if max_width > 0 else 1.0
    height_scale = max_height / height if max_height > 0 else 1.0
    scale = min(width_scale, height_scale, 1.0)
    if scale >= 1.0:
        return image

    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    weights_path = args.weights.expanduser()

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    if args.show_dimensions_mm:
        if args.mm_per_pixel is None or args.mm_per_pixel <= 0:
            raise ValueError(
                "--show-dimensions-mm requires a positive --mm-per-pixel value."
            )
    dimensions_mm_scale = args.mm_per_pixel if args.show_dimensions_mm else None

    model = YOLO(str(weights_path))
    lab_data = load_lab_data(args.data_map.expanduser(), model.names)

    if args.image is not None:
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
        active_lab_info, active_confidence = draw_detections(
            annotated_frame, result, lab_data, dimensions_mm_scale
        )
        draw_lab_panel(annotated_frame, active_lab_info, active_confidence, 0.0)

        if args.save is not None:
            cv2.imwrite(str(args.save), annotated_frame)
            print(f"Saved annotated image to: {args.save}")

        display_frame = resize_for_display(
            annotated_frame, args.display_width, args.display_height
        )
        cv2.imshow("Steel Detector - press any key to close", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

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

    window_name = "Steel Detector - press q or Esc to quit"
    previous_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            annotated_frame = frame.copy()
            active_lab_info, active_confidence = draw_detections(
                annotated_frame, results[0], lab_data, dimensions_mm_scale
            )

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            draw_lab_panel(annotated_frame, active_lab_info, active_confidence, fps)

            if writer is not None:
                writer.write(annotated_frame)

            display_frame = resize_for_display(
                annotated_frame, args.display_width, args.display_height
            )
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
