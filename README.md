# RFlash

RFlash is the official public demonstration repository for shadow reduction in ultrasound imaging using differentiable simulation and radiance field decomposition.

This repository is intentionally curated for release. It contains the code needed to load supported demo data, estimate scanner geometry, train the RFlash decomposition model, and save shadow-reduced image volumes.

## Method Overview

RFlash represents ultrasound data with two explicit parameter maps: attenuation and scatter. The differentiable renderer samples these maps along estimated ultrasound scanlines, applies exponential attenuation, time-gain compensation, and log compression, then compares the rendered slices to the observed ultrasound data. After optimization, the representation can be rendered without the attenuation term to produce a shadow-reduced volume or stack.

At a high level, the demo:

1. Loads a fetal brain 3D ultrasound volume, abdominal ultrasound image stack, or synthetic liver image stack.
2. Estimates the scanner geometry for curvilinear data from the visible fan-shaped image support.
3. Resamples curvilinear 2D images into the renderer's simulation space when needed.
4. Trains the explicit attenuation/scatter representation with a combined L2 and SSIM loss.
5. Renders the trained representation without shadowing.
6. Saves the original and shadow-reduced outputs as medical image volumes.

## Repository Structure

```text
RFlash/
├── README.md
├── LICENSE
├── requirements.txt
├── demo.py
├── data/
│   └── fetal_brain/
│       ├── fetal-brain-demo.mha
│       └── test_3d.nii.gz
└── src/
    ├── _datatypes.py
    ├── datasets.py
    ├── shadow_reduction.py
    ├── train.py
    ├── trainings_params.py
    ├── model/
    │   ├── initialization.py
    │   ├── losses.py
    │   ├── rendering.py
    │   └── representation.py
    └── utils/
        ├── geometry.py
        ├── io.py
        ├── transducer_geometry.py
        ├── transformations.py
        └── visualization.py
```

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code has been developed for Python 3.11. A CUDA or MPS accelerator is used automatically when available, otherwise the demo runs on CPU.

## Dependencies

The public demo uses:

- NumPy for array operations.
- PyTorch for optimization and differentiable rendering.
- TorchMetrics for SSIM.
- OpenCV for fan-geometry estimation and curvilinear resampling.
- MedPy for medical image I/O.
- SciPy for diagnostic morphology utilities.
- Matplotlib for image loading, overlays, and training plots.
- tqdm for training progress bars.

No tracking software, experiment database, or private infrastructure is required.

## Data

### Fetal Brain

A small fetal brain 3D ultrasound example is included under:

```text
data/fetal_brain/
```

The command line default currently points to:

```text
data/fetal_brain/test_3d.nii.gz
```

### Abdominal Ultrasound

Download the US simulation and segmentation dataset from Kaggle:

[US simulation & segmentation](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm)

Use the real ultrasound image directory as input. In the development setup this was:

```text
/home/scratch/valher/data/RFlash-demo/archive/abdominal_US/abdominal_US/RUS/images
```

The current loader reads `.jpg` files from the provided directory.

### Synthetic Liver Ultrasound

Download the synthetic testing data from the Ultra-NeRF repository:

[Synthetic liver ultrasound](https://github.com/magdalena-wysocki/ultra-nerf/tree/main/data/synthetic_testing)

The demo supports either a single `.npy` file or a directory containing compatible `.npy` files such as:

```text
images-l2.npy
images-r2.npy
```

## Running The Demo

Run the packaged fetal brain example:

```bash
python demo.py \
  --dataset fetal_brain \
  --input data/fetal_brain/test_3d.nii.gz \
  --output outputs/fetal_brain
```

Run abdominal ultrasound:

```bash
python demo.py \
  --dataset abdominal \
  --input /path/to/abdominal_US/RUS/images \
  --output outputs/abdominal
```

Run synthetic liver ultrasound:

```bash
python demo.py \
  --dataset synthetic_liver \
  --input /path/to/synthetic_testing \
  --output outputs/synthetic_liver
```

Use `--silent` to suppress non-essential plots and intermediate training-statistic visualization:

```bash
python demo.py --dataset fetal_brain --input data/fetal_brain/test_3d.nii.gz --output outputs/fetal_brain --silent
```

For abdominal stacks, the script asks how many slices to process because scanner-geometry estimation can require manual inspection when the fan edges are unclear.

## Expected Outputs

Each demo writes outputs below the directory passed with `--output`.

```text
outputs/<dataset>/
├── intermediate_images/
│   └── geometry_*.png
├── training_stats/
│   └── training_stats.png
└── shadow_removed/
    ├── original_volume.nii.gz
    └── shadow_reduced_volume.nii.gz
```

`training_stats.png` is written only when `--silent` is not used. The saved volumes contain 8-bit grayscale versions of the original and shadow-reduced data.

## Model Weights

No pretrained model weights are required. The demo trains the explicit representation from the input data using the parameters in `src/trainings_params.py`.

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

This repository is distributed under the terms described in [LICENSE](LICENSE).

## Contact

Valentin Bacher
valentin.bacher@cs.ox.ac.uk
OMNI Lab, Department of Computer Science, University of Oxford
