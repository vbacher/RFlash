# RFlash

RFlash is the public demonstration repository for shadow reduction in ultrasound imaging using differentiable simulation and radiance field decomposition.

This repository is intentionally smaller than the research codebase. It currently focuses on the first reproducible step needed by the method: loading example ultrasound data and estimating the scanner fan geometry from the image support.

## Method Overview

Curvilinear ultrasound images are formed by a virtual acoustic source and a fan-shaped field of view. RFlash uses this scanner geometry to model how ultrasound rays travel through the image volume before estimating and reducing acoustic shadows.

The current public demo estimates that geometry by:

1. Loading a 3D fetal brain volume, a stack of abdominal ultrasound images, or synthetic liver ultrasound arrays.
2. Separating the fan-shaped foreground from the dark background.
3. Tracing the left and right fan boundaries in a representative slice.
4. Fitting two boundary lines and intersecting them to estimate the virtual point source.
5. Saving an overlay so the estimated source and fan boundaries can be inspected.

For a 3D volume, the demo estimates geometry from the two middle planes of the volume.

## Repository Structure

```text
RFlash/
├── README.md
├── LICENSE
├── requirements.txt
├── demo.py
├── data/
│   └── fetal_brain/
│       └── fetal-brain-demo.mha
└── src/
    ├── __init__.py
    ├── geometry.py
    └── utils.py
```

## Installation

Create a Python environment and install the small dependency set:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The demo has been developed with Python 3.11. It uses NumPy for estimation, matplotlib for overlays and simple image loading, and MedPy for `.mha` volume I/O.

## Data

### Fetal Brain

A small fetal brain 3D ultrasound volume is included at:

```text
data/fetal_brain/fetal-brain-demo.mha
```

This is the default input for `demo.py`.

### Abdominal Ultrasound

Download the US simulation and segmentation dataset from Kaggle:

[US simulation & segmentation](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm)

Use the real ultrasound image directory as input. For example:

```text
/home/scratch/valher/data/RFlash-demo/archive/abdominal_US/abdominal_US/RUS/images
```

### Synthetic Liver Ultrasound

Download the synthetic testing data from the Ultra-NeRF repository:

[Synthetic liver ultrasound](https://github.com/magdalena-wysocki/ultra-nerf/tree/main/data/synthetic_testing)

The demo supports a directory containing files such as:

```text
images-l2.npy
images-r2.npy
```

## Running The Demo

Run the packaged fetal brain example:

```bash
python demo.py \
  --dataset fetal_brain \
  --input data/fetal_brain/fetal-brain-demo.mha \
  --output outputs/fetal_brain
```

Run abdominal ultrasound geometry estimation:

```bash
python demo.py \
  --dataset abdominal \
  --input /path/to/abdominal_US/RUS/images \
  --output outputs/abdominal
```

Run synthetic liver geometry estimation:

```bash
python demo.py \
  --dataset synthetic_liver \
  --input /path/to/synthetic_testing \
  --output outputs/synthetic_liver
```

When run from an interactive terminal, the demo shows the geometry overlay and asks whether the source was found correctly. To save overlays without opening windows or asking for confirmation, use:

```bash
python demo.py --no-confirm-geometry --no-show-overlay
```

## Expected Outputs

The output directory contains:

- `geometry.json`: estimated source location, radius range, and fan angle.
- `geometry_coronal_middle.png` and `geometry_axial_middle.png` for 3D volumes.
- `geometry_abdominal_middle.png` or `geometry_synthetic_liver_middle.png` for 2D image stacks.

The overlay marks the estimated source with a red cross and draws the fitted fan boundaries over the slice.

## Current Scope

This repository is being curated for public release. The current implementation demonstrates data loading and scanner geometry estimation. The full shadow-reduction pipeline will be added only after the public version is simplified and documented enough to be reproducible.

## Citation

If you use this code, please cite:

```bibtex
@article{bacher2026rflash,
  title = {Shadow Reduction in Ultrasound Imaging using Differentiable Simulation and Radiance Field Decomposition},
  author = {Bacher, Valentin},
  year = {2026}
}
```

Update this entry with the final publication details once available.

## License

See [LICENSE](LICENSE).

## Contact

For questions about the public RFlash demo, contact Valentin Bacher.
