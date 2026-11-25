# Quality Check Inference

Use this script to check one image for:

- Roughness error: detected roughness must match the expected roughness value.
- Dimension error: detected box size must stay inside the expected dimension range.

The script is:

```text
quality_check_inference.py
```

## Set Expected Values

Open `quality_check_inference.py` and edit these values near the top:

```python
EXPECTED_SURFACE_NUMBER = "9"
MM_PER_PIXEL = 0.05
MIN_WIDTH_MM = 45.0
MAX_WIDTH_MM = 55.0
MIN_HEIGHT_MM = 45.0
MAX_HEIGHT_MM = 55.0
```

Example:

- `EXPECTED_SURFACE_NUMBER = "9"` means only roughness 9 is accepted.
- `MIN_WIDTH_MM` and `MAX_WIDTH_MM` control the accepted width.
- `MIN_HEIGHT_MM` and `MAX_HEIGHT_MM` control the accepted height.
- `MM_PER_PIXEL` converts detected pixel size into millimeters.

## Run On An Image

Use any image from the `Data/` folder:

```bash
.venv/bin/python quality_check_inference.py --image Data/9/9_1.jpeg --device cpu
```

Another example:

```bash
.venv/bin/python quality_check_inference.py --image Data/24/24_1.jpeg --device cpu
```

The result window will show:

- `INSPECTION OK` if roughness and dimension are correct.
- `ROUGHNESS ERROR` if the detected roughness is not the expected roughness.
- `DIMENSION ERROR` if the detected box is outside the expected dimension range.

## Run On Webcam

Run live inspection with the default webcam:

```bash
.venv/bin/python quality_check_inference.py --device cpu
```

Press `q` or `Esc` to close the webcam window.

Use another webcam index if needed:

```bash
.venv/bin/python quality_check_inference.py --camera 1 --device cpu
```

Save the annotated webcam video:

```bash
.venv/bin/python quality_check_inference.py --device cpu --save quality_webcam_output.mp4
```

The webcam view uses the same expected roughness, dimension limits, and `MM_PER_PIXEL` values that are set near the top of `quality_check_inference.py`.

## Save Output Image

To save the annotated result:

```bash
.venv/bin/python quality_check_inference.py --image Data/9/9_1.jpeg --device cpu --save quality_output.jpg
```

The saved image will include the detection box, roughness status, dimension status, and measured box size in mm.

## Change Model Or Confidence

Use another model weights file:

```bash
.venv/bin/python quality_check_inference.py --image Data/9/9_1.jpeg --weights runs/detect/steel_detector4/weights/best.pt --device cpu
```

Increase confidence threshold:

```bash
.venv/bin/python quality_check_inference.py --image Data/9/9_1.jpeg --conf 0.40 --device cpu
```

## Calibration

The dimension check depends on `MM_PER_PIXEL`.

If the script shows `250 mm`, but the real object is `50 mm`, calculate:

```text
new MM_PER_PIXEL = old MM_PER_PIXEL * real size / shown size
                 = 0.25 * 50 / 250
                 = 0.05
```

Then set:

```python
MM_PER_PIXEL = 0.05
```

## Useful Image Paths

Examples from this project:

```text
Data/9/9_1.jpeg
Data/9/9_2.jpeg
Data/14/14_1.jpeg
Data/17/17_1.jpeg
Data/24/24_1.jpeg
```

Use the image path that matches the sample you want to inspect.
