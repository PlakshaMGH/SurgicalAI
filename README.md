# Plaksha - MGH: AI in Thoracic Surgery

## Getting Started

Install [uv(python package manager)](https://github.com/astral-sh/uv)
```
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

Install dependencies
```
$ uv venv --python=3.12

# activate the venv
$ source .venv/bin/activate

# install dependencies
$ uv pip install -r requirements.txt
```

## Tasks
* Video Segmentation
* Action Recognition in Surgical Videos
* Phase Recognition in Surgical Videos

## Data
* EndoVis 17 - Robotic Instrument Segmentation
  * Binary Masks
  * Part Types Masks (Shaft, Wrist, Clasper)
  * Instrument Type Masks (Bipolar Forceps, Prograsp Forceps, Large Needle Driver, Vessel Sealer, Grasping Retractor, Monopolar Curved Scissors)
  * Combined Type Masks (both Part and Instrument Masks)
* EndoVis18 - Robotic Instrument Segmentation
  * Binary Masks
  * Part Type Masks (Shaft, Clasper, Wrist)
* Right Lower Lobe
  * Pulmonary Artery
  * Inferior Pulmonary Vein
* Left Upper Lobe
  * Pulmonary Artery
  * Inferior Pulmonary Vein