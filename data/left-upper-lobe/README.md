# Left Upper Lobe

This folder contains the data for the left upper lobe from Massachusetts General Hospital.

## Data

The data is stored in the B2 cloud storage. We use the `rclone` tool to download the data.

To download the data, run the `get_b2_data.sh` script by changing the bucket and local path.

```bash
./get_b2_data.sh
```

## Structure

The data is organized by patient and then by study.

```
left-upper-lobe/
├── pa/
│   ├── data/ # original data received from MGH
│   │   ├── Patient01_AnnotationComplete/
│   │   │   ├── lul-arterial-dissection-pt1-frames0000-1000/
│   │   │   │   ├── instances_default.json
│   │   │   ├── PA-2401.mp4
│   │   ├── Patient02_AnnotationComplete/
│   │   │   ├── lul-arterial-dissection-pt2-frames1600-1749/
│   │   │   │   ├── instances_default.json
│   │   │   ├── lul-arterial-dissection-pt2-frames1750-1900
│   │   │   │   ├── instances_default.json
│   │   │   ├── Untitled Project.mov
│   │   ├── Patient03_AnnotationComplete/
│   │   │   ├── lul - arterial dissection pt 3 - annotations-frames790-1090/
│   │   │   │   ├── instances_default.json
│   │   │   ├── lul - arterial dissection pt 2.mov
|   ├── frames/ # processed videos
|   |   ├── p01/
|   |   |   ├── frame_000000.png
|   |   |   ├── frame_000001.png
|   |   |   ├── ...
|   |   |   ├── frame_000999.png
|   |   ├── p02/
|   |   |   ├── frame_001600.png
|   |   |   ├── frame_001601.png
|   |   |   ├── ...
|   |   |   ├── frame_001899.png
|   |   ├── p03/
|   |   |   ├── frame_000790.png
|   |   |   ├── frame_000791.png
|   |   |   ├── ...
|   |   |   ├── frame_001089.png
|   ├── masks/ # processed annotations
|   |   ├── p01/
|   |   |   ├── frame_000000.png
|   |   |   ├── frame_000001.png
|   |   |   ├── ...
|   |   |   ├── frame_000999.png
|   |   ├── p02/
|   |   |   ├── frame_001600.png
|   |   |   ├── frame_001601.png
|   |   |   ├── ...
|   |   |   ├── frame_001899.png
|   |   ├── p03/
|   |   |   ├── frame_000790.png
|   |   |   ├── frame_000791.png
|   |   |   ├── ...
|   |   |   ├── frame_001089.png
├── pv/ # Empty for now
```