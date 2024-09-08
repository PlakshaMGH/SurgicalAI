#! /bin/bash

# check if rclone is installed
if ! command -v rclone &> /dev/null
then
    echo "rclone could not be found, please install rclone first"
    exit
fi

# check if remote is already configured
if ! rclone lsd remote: > /dev/null 2>&1
then
    echo "Remote is not configured, please configure rclone first"
    exit
fi

echo "Downloading data from B2"
rclone copy remote:right-lower-lobe/ ./data/right-lower-lobe --progress --transfers 32