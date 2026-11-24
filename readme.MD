# DataMatrix Fixer

Reads damaged, worn or hard-to-read **DataMatrix (ECC 200)** codes, including laser-marked ones, from one or many handheld photos.

Three methods work together:

1. **Built-in decoder (no candidate list needed).** The image is perspective-corrected and rotated using the finder pattern. The module grid is sampled, the ECC 200 placement is applied, and Reed-Solomon error correction runs on the result. Modules the program is unsure about are treated as *erasures*, which lets it correct roughly twice as many damaged codewords as an ordinary decoder.
2. **Generate-and-compare (optional).** If you supply a list of possible serial numbers, each one is encoded as a DataMatrix and compared with the evidence combined from all images. This still works when the code is too damaged for error correction.
3. **Template fit (optional, for small / blurry / damaged images).** When the image is too low-resolution to read the module grid directly (e.g. 3-4 pixels per module), each candidate is rendered as a template and fitted straight into the image by optimizing a perspective transform to maximize correlation. It can ignore one damaged edge strip (for example a few columns that were marked shifted). It runs automatically when a candidate list is given and the built-in decoder fails (`--fit auto`), or always with `--fit on`.

## Features

- Symbol size by parameter (`--size 20`, `8x18`, `18,20,22`) or **automatic** (`--size auto`, default)
- All standard ECC 200 sizes, square and rectangular, including multi-region symbols
- Perspective correction with line-fitted corners (robust against missing corner modules and glow around bright marks)
- Any rotation, inverted contrast (bright marks on dark background), uneven lighting
- Multi-image fusion, weighted by image quality
- Decodes ASCII, C40, Text, X12, EDIFACT and Base256 data
- False-positive protection: reserved error-correction margin and plausibility check on the decoded text
- Optional fast path through libdmtx / zxing-cpp

## Installation

```bash
py -m pip install opencv-python numpy setuptools scipy
py -m pip install pylibdmtx zxing-cpp      # optional
```

`scipy` is needed for the template fit. `pylibdmtx` and `zxing-cpp` are optional: the built-in decoder, the template fit (ASCII encoding) and the candidate comparison all work without them. `pylibdmtx` only adds other encoding schemes (C40, Text, X12, EDIFACT, Base256) for candidates.

Windows notes:
- `No module named 'distutils'` (Python 3.12+): `py -m pip install setuptools`
- DLL errors from `pylibdmtx`: install *Microsoft Visual C++ Redistributable (x64)*
- The pip package is `opencv-python`, not `cv2`

## Usage

```bash
# Decode all images in a folder and combine them into one result, size detected automatically
py datamatrixFixer.py --images images/

# Known size
py datamatrixFixer.py --size 20 --images images/

# Each image separately
py datamatrixFixer.py --size 20 --images images/ --each

# With a candidate list (one serial per line, spaces preserved exactly)
py datamatrixFixer.py serials.txt --size 18,20 --images images/

# Live camera (r = reset, q = quit)
py datamatrixFixer.py --size 20 --camera 0
```

Example output:

```
ser0001.png [20x20]: quality 7.77  own: 'PCBA;200043-C;DAR;2635;0001'

=== Combined result ===
Size 20x20 (1 images):
  Own decoder: 'ABCD;12345678-C;ABC;2635;0001'  (RS: 0 errors corrected, 0 codewords erased; PAD)
RESULT: 'ABCD;12345678-C;ABC;2635;0001'  [size 20x20, own decoder]
```

(The program's console messages are in Finnish.)

### Options

| Option | Description |
|---|---|
| `serials` | Optional text file with candidate serial numbers, one per line. Do not trim spaces. |
| `--size` | `20`, `16x16`, `8x18`, a list `18,20,22`, or `auto` (default). |
| `--images` | Folder or glob (e.g. `images/*.png`). |
| `--camera` | Camera index for live mode. |
| `--each` | Process every image independently instead of combining them. |
| `--no-direct` | Skip the libdmtx/zxing fast path. |
| `--scheme` | Restrict candidate encoding schemes (repeatable), e.g. `--scheme Ascii`. |
| `--dp-min` | Minimum image quality (default 1.5). Weaker images are ignored. |
| `--max-err`, `--min-margin` | Acceptance thresholds for the candidate comparison. |
| `--fit` | `auto` (default), `on` or `off`. Template fit for small, blurry or damaged images. Needs a candidate list. |
| `--fit-min`, `--fit-margin` | Template-fit acceptance: minimum correlation (default 0.55) and minimum gap to the runner-up (0.08). |

## How the result is judged

- **Own decoder:** a result is accepted only if Reed-Solomon verification passes with spare correction capacity left over, and the decoded text looks sensible. If you supplied candidates, the output also says whether the decoded text is in the list.
- **Candidate comparison:** the best candidate must have a low mismatch and a clear margin over the runner-up, otherwise it is reported as uncertain.

## Tips

- Take several photos (5-15) from slightly different angles and lighting, and let the program combine them.
- The code should be roughly 100 px or more across in the image. For a 3 mm code, shoot close-up or macro.
- Lighting matters more than software. For laser marks, diffuse dome light or low-angle light helps a lot.
- If you know the size, pass `--size`. It is faster and avoids rare misdetections.
- When damage is so heavy that the built-in decoder fails, supply a candidate list.

## Verification status

- Tested on a real 20x20 code: decoded correctly, including after synthetic rotation, perspective, uneven lighting, noise, blur, scratches and inversion.
- Template fit: on a real, small (about 87x98 px, ~3.6 px per module) 18x18 photo with a displaced edge strip, the confirmed string `200043-c 2635  DA 0010` scored a correlation of 0.79 against at most 0.53 for 120 other candidates. On a heavily damaged synthetic image (a large block blacked out) the built-in decoder failed, while the template fit still picked the right serial.
- A second small photo (`kuva1_A`) did not match any tested candidate with confidence (best 0.50, margin below the threshold), so the program reports no result there rather than guessing.
- Reed-Solomon correction tested with simulated damage (up to 7 wrong codewords, or 12-14 uncertain ones).
- No false positives on 450 random-noise trials across all sizes.
- Other symbol sizes use the ISO 16022 tables and the standard placement algorithm. The tables are internally consistent, but only 20x20 has been verified against a real code.
- The candidate comparison (requires `pylibdmtx`) and live camera mode could not be tested in the development environment.

## Limitations

- Template fit is slow: roughly 15-40 s per image for ~100 candidates (it fits every candidate separately). Keep the candidate list as specific as you can, and use `--size` and `--scheme` to narrow the search.
- Template fit needs a clearly better candidate than the others. If the true string is not in the list (for example the number of spaces differs), no result is reported.

- Large confidently-wrong damage (e.g. an area blacked out) cannot be detected as erasures; use the candidate list there.
- Code localization is contour-based and may fail on cluttered backgrounds. Crop closer if needed.
- Dot-peen marks would benefit from a dilation step (not included).

## License

Dependencies: OpenCV (Apache 2.0), NumPy (BSD), libdmtx/pylibdmtx (BSD), zxing-cpp (Apache 2.0).
