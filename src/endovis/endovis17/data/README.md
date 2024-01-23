The data folder structure.
```
> data
    > frames
        > endo17_test_frames
            > instrument_dataset_01
                - frame225.png
                - frame226.png
                - ...
                - frame299.png
            > instrument_dataset_02
            ...
            > instrument_dataset_10
        > endo17_train_frames
            > instrument_dataset_01
                - frame000.png
                - frame001.png
                - ...
                - frame224.png
            > instrument_dataset_02
            ...
            > instrument_dataset_08
    > masks
        > endo17_test_masks
            > instrument_dataset_01
            > instrument_dataset_02
            ...
            > instrument_dataset_10
        > endo17_train_masks
            > instrument_dataset_01
            > instrument_dataset_02
            ...
            > instrument_dataset_08
    > yolo_labels
        > test
            > instrument_dataset_01
                - frame225.txt
                - frame226.txt
                - ...
                - frame299.txt
            > instrument_dataset_02
            ...
            > instrument_dataset_10
        > train
           > instrument_dataset_01
                - frame000.txt
                - frame001.txt
                - ...
                - frame224.txt
            > instrument_dataset_02
            ...
            > instrument_dataset_08
```

### Backblaze

* bucketname: endovis
* bucketpath: endovis/endo17
* endpoint url: https://s3.us-west-004.backblazeb2.com

