#!/bin/bash

# Update packages
sudo apt-get update \
    && sudo apt-get install -y p7zip-full unzip

# Install python virtual environment
python3 -m pip install virtualenv==20.10.0

# Createand activate virtual environment
if [ ! -d ".venv" ]
then
    python3 -m venv .venv
fi
source ./.venv/bin/activate

# Install required packages
python3 -m pip install -r requirements.txt

./download_dataset.sh
