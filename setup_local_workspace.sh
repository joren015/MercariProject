#!/bin/bash

# Update packages
sudo apt-get update \
    && sudo apt-get install -y p7zip-full unzip

# Install python virtual environment
python -m pip install virtualenv==20.10.0

# Createand activate virtual environment
python -m venv .venv
source ./.venv/bin/activate

# Install required packages
python -m pip install -r requirements.txt

./download_dataset.sh

