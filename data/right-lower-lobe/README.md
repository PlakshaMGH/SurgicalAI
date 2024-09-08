# Right Lower Lobe

This folder contains the data for the right lower lobe from Massachusetts General Hospital.

## Data

The data is stored in the B2 cloud storage. We use the `rclone` tool to download the data.

To download the data, run the `get_b2_data.sh` script by changing the bucket and local path.

```bash
./get_b2_data.sh
```

## Structure

The data is organized by patient and then by study.

```
right-lower-lobe/
├── pa/
|   ├── frames/ # processed videos
|   |   ├── p01/
|   |   |   ├── frame_001401.png
|   |   |   ├── frame_001402.png
|   |   |   ├── ...
|   |   |   ├── frame_002400.png
|   |   ├── p02/
|   |   |   ├── frame_000350.png
|   |   |   ├── frame_000351.png
|   |   |   ├── ...
|   |   |   ├── frame_000650.png
|   |   ├── ...
|   |   ├── p20/
|   |   |   ├── frame_000000.png
|   |   |   ├── frame_000001.png
|   |   |   ├── ...
|   |   |   ├── frame_000300.png
|   ├── masks/ # processed annotations
|   |   ├── p01/
|   |   |   ├── frame_001400.png
|   |   |   ├── frame_001401.png
|   |   |   ├── ...
|   |   |   ├── frame_002400.png
|   |   ├── p02/
|   |   |   ├── frame_000350.png
|   |   |   ├── frame_000351.png
|   |   |   ├── ...
|   |   |   ├── frame_000650.png
|   |   ├── ...
|   |   ├── p20/
|   |   |   ├── frame_000000.png
|   |   |   ├── frame_000001.png
|   |   |   ├── ...
|   |   |   ├── frame_000300.png
├── pv/
|   ├── data/ # original data received from MGH
|   |   |   ├── Patient1_RLL_IPV_AnnotationComplete/
|   |   |   |   ├── inferior-pulmonary-vein-ms-RLL-1k+.mp4
|   ├── frames/ # processed videos
|   |   ├── p01/
|   |   |   ├── frame_000040.png
|   |   |   ├── frame_000041.png
|   |   |   ├── ...
|   |   |   ├── frame_001090.png
|   ├── masks/ # processed annotations
|   |   ├── p01/
|   |   |   ├── frame_000040.png
|   |   |   ├── frame_000041.png
|   |   |   ├── ...
|   |   |   ├── frame_001090.png

```