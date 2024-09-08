# EndoVis 2017

This folder contains the data for the EndoVis 2017 challenge.

## Data

The data is stored in the B2 cloud storage. We use the `rclone` tool to download the data.

To download the data, run the `get_b2_data.sh` script by changing the bucket and local path. Also, make sure to exclude the original data downloaded folder. (folder: OG)

```bash
./get_b2_data.sh
```

## Structure

```
endo17/
├── data/
│   ├── frames/
│   │   ├── train/
│   │   │   ├── instrument_dataset_01
│   │   │   │   ├── frame000.png
│   │   │   │   ├── ...
│   │   │   │   ├── frame224.png
│   │   │   ├── instrument_dataset_02
│   │   │   ├── ...
│   │   │   ├── instrument_dataset_08
│   │   ├── test/
│   │   │   ├── instrument_dataset_01
│   │   │   │   ├── frame225.png
│   │   │   │   ├── ...
│   │   │   │   ├── frame299.png
│   │   │   ├── instrument_dataset_02
│   │   │   ├── ...
│   │   │   ├── instrument_dataset_09
│   │   │   │   ├── frame000.png
│   │   │   │   ├── ...
│   │   │   │   ├── frame299.png
│   │   │   ├── instrument_dataset_10
│   ├── masks/
│   │   ├── train/
│   │   │   ├── binary_masks/
│   │   │   │   ├── instrument_dataset_01
│   │   │   │   │   ├── frame000.png
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── frame224.png
│   │   │   │   ├── instrument_dataset_02
│   │   │   │   ├── ...
│   │   │   │   ├── instrument_dataset_08
│   │   │   │   │   ├── frame000.png
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── frame224.png
│   │   │   ├── comb_masks/
│   │   │   │   ├── ...
│   │   │   ├── part_masks/
│   │   │   │   ├── ...
│   │   │   ├── type_masks/
│   │   │   │   ├── ...
│   │   ├── test/
│   │   │   ├── binary_masks/
│   │   │   │   ├── instrument_dataset_01
│   │   │   │   │   ├── frame225.png
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── frame299.png
│   │   │   │   ├── instrument_dataset_02
│   │   │   │   ├── ...
│   │   │   │   ├── instrument_dataset_09
│   │   │   │   │   ├── frame000.png
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── frame299.png
│   │   │   │   ├── instrument_dataset_10
│   │   │   ├── comb_masks/
│   │   │   │   ├── ...
│   │   │   ├── part_masks/
│   │   │   │   ├── ...
│   │   │   ├── type_masks/
│   │   │   │   ├── ...
├── weights/
│   ├── binary
│   │   ├── XMem_Endo17Bin_ColorJitter.pth
│   │   ├── ...
│   │   ├── XMem_Endo17Bin_RandResize.pth
│   ├── comb # empty for now
│   ├── part # empty for now
│   ├── type # empty for now
```