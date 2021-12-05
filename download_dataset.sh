#!/bin/bash

DATA_FOLDER=data

if [ ! -d "$DATA_FOLDER" ] 
then
    mkdir $DATA_FOLDER
fi

if [ ! -f "$DATA_FOLDER/mercari-price-suggestion-challenge.zip" ]
then
    kaggle competitions download -p $DATA_FOLDER -c mercari-price-suggestion-challenge
fi

unzip -d $DATA_FOLDER -o $DATA_FOLDER/mercari-price-suggestion-challenge.zip
unzip -d $DATA_FOLDER -o $DATA_FOLDER/test_stg2.tsv.zip
unzip -d $DATA_FOLDER -o $DATA_FOLDER/sample_submission_stg2.csv.zip
7za e $DATA_FOLDER/train.tsv.7z -aoa -o$DATA_FOLDER
7za e $DATA_FOLDER/test.tsv.7z -aoa -o$DATA_FOLDER
7za e $DATA_FOLDER/sample_submission.csv.7z -aoa -o$DATA_FOLDER
