# EndoVis 2018

This folder contains the data for the EndoVis 2018 challenge.

## Data

The data is stored in the B2 cloud storage. We use the `rclone` tool to download the data.

To download the data, run the `get_b2_data.sh` script by changing the bucket and local path. Also, make sure to exclude the original data downloaded folder. (folder: OG)

```bash
./get_b2_data.sh
```

## Structure

```
endo18/
├── data/
│   ├── frames/
│   │   ├── seq_01
│   │   │   ├── frame000.png
│   │   │   ├── ...
│   │   │   ├── frame148.png
│   │   ├── seq_02
│   │   ├── ...
│   │   ├── seq_20
│   ├── masks/
│   │   ├── all_binary_masks/
│   │   │   ├── seq_01
│   │   │   │   ├── frame000.png
│   │   │   │   ├── ...
│   │   │   │   ├── frame148.png
│   │   │   │   ├── seq_02
│   │   │   │   ├── ...
│   │   │   │   ├── seq_20
│   │   ├── all_masks/
│   │   │   ├── ...
│   │   ├── binary_masks/
│   │   │   ├── ...
│   │   ├── part_masks/
│   │   │   ├── ...
```

