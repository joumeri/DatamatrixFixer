# DataMatrix Fixer

Reads damaged, worn or hard-to-read **laser-marked DataMatrix codes** (ECC 200) by matching them against a list of known serial number candidates.

Instead of forcing a decoder to succeed on a bad image, the program turns the problem around:

1. Each image is perspective-corrected and rotated to the correct orientation.
2. The module grid is sampled and every module gets a confidence value.
3. Evidence from **multiple handheld images** is combined module by module.
4. Every candidate serial number is encoded as a DataMatrix and compared to the combined evidence. The best match wins, but only if it is clearly better than the runner-up.

This works well when the set of possible serial numbers is small (tens to a few hundred), and it tolerates missing or destroyed modules far better than a normal decoder.

## Features

- Perspective correction and automatic orientation using the known finder pattern (solid "L") and clock track
- Illumination-independent thresholding (threshold is measured from the finder/clock modules of each image)
- Handles inverted contrast (light marks on dark background) automatically
- Small grid-alignment search (sub-module offset and scale) to compensate for handheld shots
- Multi-image fusion, weighted by image quality
- Square and rectangular symbols (e.g. `18x18`, `8x18`, `16x48`), one or several sizes at once
- All encoding schemes tried (ASCII, C40, Text, X12, EDIFACT, Base256)
- Fast path: ordinary decoding with libdmtx / zxing-cpp is tried first
- Confidence reporting: results are rejected as uncertain if the match is poor or the margin to the second-best candidate is small, so a wrong serial number does not slip through silently

## Installation

```bash
py -m pip install opencv-python numpy pylibdmtx zxing-cpp
```

`zxing-cpp` is optional (used as a fallback decoder).

### Windows notes

- **`ModuleNotFoundError: No module named 'distutils'`** (Python 3.12+): `pylibdmtx` still imports `distutils`, which was removed from the standard library. Fix:
  ```bash
  py -m pip install setuptools
  ```
- **DLL load errors** when importing `pylibdmtx`: install the *Microsoft Visual C++ Redistributable (x64)*.
- The package to install is `opencv-python`, not `cv2`.

Verify the installation:

```bash
py -c "import cv2, numpy, pylibdmtx.pylibdmtx; print(cv2.__version__)"
```

## Usage

### 1. Candidate list

Create a text file with **one serial number per line**, e.g. `serials.txt`:

```
AB   123.456:78
AB   123.457:78
```

Spaces are preserved exactly (only the line break is removed). `"AB   123"` and `"AB  123"` are different candidates, so make sure the number of spaces, and any trailing spaces, is correct.

### 2. Run

```bash
# All images in a folder -> one combined result
py datamatrixFixer.py serials.txt --size 18 --images images/

# Each image processed separately
py datamatrixFixer.py serials.txt --size 18 --images images/ --each

# Several possible symbol sizes (best match wins)
py datamatrixFixer.py serials.txt --size 16,18,20 --images images/

# Rectangular symbol
py datamatrixFixer.py serials.txt --size 8x18 --images images/

# Live camera (r = reset accumulated evidence, q = quit)
py datamatrixFixer.py serials.txt --size 18 --camera 0
```

### Options

| Option | Description |
|---|---|
| `--size` | Symbol size in modules: `18`, `16x16`, `8x18`, or a comma-separated list. Default `18`. |
| `--images` | Folder or glob pattern (e.g. `images/*.jpg`). |
| `--camera` | Camera index for live mode. |
| `--each` | Treat every image independently instead of combining them. |
| `--no-direct` | Skip the normal decoding fast path. |
| `--scheme` | Restrict encoding schemes (repeatable), e.g. `--scheme Ascii`. Default: all. |
| `--dp-min` | Minimum image quality (default `1.5`). Weaker images are discarded. |
| `--max-err` | Maximum accepted mismatch of the best candidate (default `0.20`). |
| `--min-margin` | Minimum gap to the second-best candidate (default `0.03`). |

### Output

```
=== Combined result ===
Size 18x18 (8 images):
  0.041  'AB   123.456:78'  (Ascii)
  0.212  'AB   123.457:78'  (Ascii)
  -> 'AB   123.456:78'  OK: error 0.041, margin 0.171, scheme Ascii
```

- **error**: weighted fraction of mismatching modules (0 = perfect, ~0.5 = random)
- **margin**: difference to the second-best candidate
- A result is flagged as *uncertain* when the margin is too small, and as a poor match when the error is too high. Check those manually.

## Tips for good results

- **Take several images (5-15)** from slightly different angles and lighting. Reflections land in different places in each shot, and fusion removes them.
- **Fill the frame**: the code should be at least roughly 100-150 px across. A 3 mm code needs close-up or macro shooting.
- **Restrict the scheme** when you know it. Take one readable code from the same batch, see which scheme matches in the output, then use e.g. `--scheme Ascii`. This is faster and reduces false matches.
- **Restrict the size**. Each additional size adds processing time and a small chance of a false match.
- **Tune the thresholds** (`--max-err`, `--min-margin`, `--dp-min`) using ten or so known codes: see at which error correct matches arrive and how close wrong ones get.
- **Lighting matters more than software.** For laser marks on metal, diffuse dome light or low-angle/dark-field lighting makes a big difference.

## Limitations

- Developed and syntax-checked, but **not yet tuned on real-world image sets**. Thresholds will need adjusting to your images.
- Code localization is contour-based. It can fail on cluttered backgrounds. Crop the image closer to the code if detection fails.
- Dot-peen / dotted marks would benefit from a dilation step before sampling (not included by default).
- Strong illumination gradients (shiny metal) would benefit from per-module local thresholding.
- Candidates that do not fit the chosen symbol size are skipped (the program reports how many).

## How it works (short version)

| Step | What happens |
|---|---|
| Localize | Adaptive threshold + contours, both polarities, squareness filter |
| Rectify | Perspective warp to a fixed grid (16 px per module) |
| Align | Search over sub-module offsets and scale; 4 rotations; normal/inverted contrast; choose the best fit to the known finder + clock pattern |
| Threshold | Dark/light levels measured from the finder and clock modules; each module becomes an evidence value in [-1, 1] |
| Fuse | Quality-weighted average of evidence over all images |
| Match | Candidates encoded with libdmtx in all schemes; weighted mismatch against the evidence; best candidate + margin |

## License

Dependencies: OpenCV (Apache 2.0), NumPy (BSD), libdmtx / pylibdmtx (BSD), zxing-cpp (Apache 2.0).
