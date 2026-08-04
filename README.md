# RFlash

### [Project Page](https://vbacher.github.io/RFlash-ultrasound/) | [Demo] | [Paper]

RFlash is the official public demonstration repository for shadow reduction in ultrasound imaging using differentiable simulation and radiance field decomposition.
For a GUI please use app.py, for a CLI interface use demo.py. The backend is the same.

This repository is intentionally curated for release. It contains the code needed to load supported demo data, estimate scanner geometry, train the RFlash decomposition model, and save shadow-reduced image volumes.

## TL;DR quickstart

To setup the python environment and run the fetal brain demo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
python demo.py \
  --dataset fetal_brain \
  --input data/fetal_brain/test_3d.nii.gz \
  --output outputs/fetal_brain
```

To launch the web interface on your local machine, run:

```bash
python app.py
```

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
├── app.py
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

The Gradio app listens on `0.0.0.0` by default so it can be reached through SSH port forwarding or deployed to Hugging Face Spaces without code changes.

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
- Gradio for the optional browser-based interface.

No tracking software, experiment database, or private infrastructure is required.

## Data

### Fetal Brain

A small fetal brain 3D ultrasound example is included under:

```text
data/fetal_brain/
```

### Abdominal Ultrasound

Download the US simulation and segmentation dataset from Kaggle:

[US simulation & segmentation](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm)

Use the real ultrasound image (RUS) directory as input.

The current loader reads `.jpg` files from the provided directory.

### Synthetic Liver Ultrasound

Download the synthetic testing data from the Ultra-NeRF repository:

[Synthetic liver ultrasound](https://github.com/magdalena-wysocki/ultra-nerf/tree/main/data/synthetic_testing)

The demo supports either a single `.npy` file or a directory containing compatible `.npy` files such as:

```text
images-l2.npy
images-r2.npy
```

## Running The Gradio Interface

Start the web application with:

```bash
python app.py
```

For information on parameters, run:

```bash
python app.py --help
```

By default the app listens on `0.0.0.0:7860` and writes outputs to:

```text
outputs/gradio/
```

On a remote Linux machine, forward the port from your local computer:

```bash
ssh -L 7860:localhost:7860 user@remote-host
```

Then open:

```text
http://localhost:7860
```

The app supports two ways to provide data:

- Upload one or more files through the browser.
- Enter a server-side path when the data already exists on the remote machine.

The interface asks whether the data is a 3D volume or 2D image data. For 2D data it asks whether the probe is linear or curvilinear. Curvilinear inputs require scanner geometry estimation; the app displays geometry overlays and lets the user accept or reject candidate slices before training.

Accepted inputs in the web interface:

- 3D volume: one `.mha`, `.nii`, or `.nii.gz` file.
- 2D stack, linear probe: one or more image files, a stack saved as `.mha`, `.nii`, or `.nii.gz`, or synthetic-liver style `.npy` input.
- 2D stack, curvilinear probe: one or more image files, or a stack saved as `.mha`, `.nii`, or `.nii.gz`.

The app also allows the user to keep the default training parameters or override the main settings manually before running RFlash. During training, the Gradio progress bar is driven by training iterations rather than estimated seconds, and the loss, L2, and SSIM curves update live in the interface.

Sharing is configurable and is disabled by default. To enable a Gradio share link:

```bash
python app.py --share
```

or:

```bash
RFLASH_GRADIO_SHARE=1 python app.py
```

Use `--no-share` to override the environment variable. Use `--server-port`, `--server-name`, and `--output` to configure deployment details:

```bash
python app.py --server-name 0.0.0.0 --server-port 7860 --output outputs/gradio
```

The Gradio path prefers CUDA when available. If CUDA is not available it falls back to MPS when available, then CPU.

Processed output formats:

- Volumes can be exported as `.nii.gz` or `.mha`.
- 2D outputs can be exported as `.nii.gz`, `.mha`, or a zip archive of `.png` or `.jpg` slices.
  
### Using the graphical user interface on a local machine

If you use the app on a local machine with a GPU, please tick the box `advanced settings`. Instead of uploading the data you can use absolute file paths for input and output.


## Running The CLI Demo

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

The command line interface and the Gradio application both call the shared inference routines in `src/inference.py`. This keeps a single implementation of the RFlash training and rendering pipeline.

## Local Testing

After installing dependencies, verify the entry points:

```bash
python demo.py --help
python app.py --help
```

Run a small CLI smoke test with the packaged fetal brain example:

```bash
python demo.py \
  --dataset fetal_brain \
  --input data/fetal_brain/test_3d.nii.gz \
  --output outputs/local_test \
  --silent
```

For the web interface, launch:

```bash
python app.py --server-port 7860
```

Open `http://localhost:7860`, select the packaged fetal brain volume, estimate and confirm geometry, then run RFlash. On a remote machine, use SSH port forwarding as shown above.

## Hugging Face Spaces Deployment

Create a Gradio Space and upload this repository. The Space should install `requirements.txt` and run:

```bash
python app.py --server-name 0.0.0.0 --server-port 7860
```

For Spaces, do not enable `share=True`; the Space itself provides the public URL. Keep any example data small enough for public distribution, and ask users to upload or provide paths for larger abdominal or synthetic liver datasets.

## Scanner Geometry Estimation

For our method to work, the scanner geometry must be known, especially for curvilinear and 3D scanners. The geometry is estimated by trying to identify the fan edges in the images and tracing them back to a point source.

This is done automatically, but it can fail, especially when the edges are not visible because of coupling artifacts. For the suggested datasets, this is most relevant for the abdominal dataset. As such, the user is expected to inspect and validate the estimated geometry.

Criteria:
- The point source should be located in a sensible position above the fan.
- The yellow circle parts should lie close to the upper and lower edges of the fan.
- The blue lines on each side should point in the direction of the fan edge. **It is acceptable if they do not trace the entire fan edge.** They are only used to indicate direction, not length.

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
