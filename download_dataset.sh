#!/bin/bash

if [ ! -d "data" ] 
then
    mkdir data
fi

if [ ! -f "data/mercari-price-suggestion-challenge.zip" ]
then
    kaggle competitions download -p data -c mercari-price-suggestion-challenge
fi

unzip -d data -o data/mercari-price-suggestion-challenge.zip
unzip -d data -o data/test_stg2.tsv.zip
7za e data/train.tsv.7z -aoa -odata
7za e data/test.tsv.7z -aoa -odata
