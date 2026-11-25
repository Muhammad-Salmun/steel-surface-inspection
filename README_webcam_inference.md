# Steel Detector Webcam Inference

This script runs inference with the trained YOLO model:

```text
runs/detect/steel_detector4/weights/best.pt
```

It can run on a webcam or on a single test image. Detected YOLO classes such as `roughness_14` are mapped to the lab data files in:

```text
data output map/
```

The output window shows the detected surface number and measured lab values such as `Ra`, `Rq`, `Rz`, `Cut-Off`, `Filter`, and `Standard`.

## Requirements

Use the project virtual environment:

```bash
.venv/bin/python webcam_inference.py --help
```

If dependencies are missing, install them:

```bash
.venv/bin/pip install -r requirements.txt
```

The `.xls` lab data files require LibreOffice for conversion:

```bash
sudo apt install libreoffice
```

The `.xlsx` files are read directly by the script.

## Webcam Inference

Run webcam detection:

```bash
.venv/bin/python webcam_inference.py --device cpu
```

Press `q` or `Esc` to close the webcam window.

Use another webcam index if needed:

```bash
.venv/bin/python webcam_inference.py --camera 2 --device cpu
```

Change the confidence threshold:

```bash
.venv/bin/python webcam_inference.py --conf 0.35 --device cpu
```

Show detected box dimensions in millimeters:

```bash
.venv/bin/python webcam_inference.py --device cpu --show-dimensions-mm --mm-per-pixel 0.05
```

`--mm-per-pixel` must match your camera setup. For example, if a 100 mm reference object appears as 400 pixels wide in the image, use `--mm-per-pixel 0.25`.

## Image Testing

Run inference on a single image:

```bash
.venv/bin/python webcam_inference.py --image Data/12/12_2.jpeg --device cpu --show-dimensions-mm --mm-per-pixel 0.05
```

Press any key to close the image window.

Save the annotated result:

```bash
.venv/bin/python webcam_inference.py --image dataset/images/val/24_1_jpeg.rf.iPvchfhXwb7U4yCKotF8.jpeg --device cpu --save test_output.jpg
```

## Display Size

Large images are scaled down for display by default. The saved output remains full resolution.

Use a smaller preview:

```bash
.venv/bin/python webcam_inference.py --image your_image.jpg --device cpu --display-width 900 --display-height 600
```

Use original preview size:

```bash
.venv/bin/python webcam_inference.py --image your_image.jpg --device cpu --display-width 0 --display-height 0
```

## Saving Webcam Output

Save the annotated webcam video:

```bash
.venv/bin/python webcam_inference.py --device cpu --save webcam_output.mp4
```

## Lab Data Mapping

Each detected class maps to a spreadsheet by surface number:

```text
roughness_1  -> data output map/1.xls
roughness_14 -> data output map/14.xlsx
roughness_24 -> data output map/24.xlsx
```

The script reads values from those files and replaces the raw class name with useful lab data on screen.

## Common Options

```bash
--weights PATH        Model weights path
--camera INDEX       Webcam index, default 0
--image PATH         Test on one image instead of webcam
--conf VALUE         Confidence threshold, default 0.25
--imgsz SIZE         YOLO inference image size, default 640
--device DEVICE      Use cpu, 0, cuda:0, etc.
--save PATH          Save annotated image or video
--data-map PATH      Lab data folder, default "data output map"
--display-width N    Max preview width, default 1100
--display-height N   Max preview height, default 720
--show-dimensions-mm Show detected box dimensions in millimeters
--mm-per-pixel VALUE  Camera scale used for dimensions, in mm per pixel
```

## Notes

If you see a CUDA warning, run with CPU:

```bash
.venv/bin/python webcam_inference.py --device cpu
```

The script displays micrometers as `um` because OpenCV's default text renderer does not draw the `µ` symbol correctly.
