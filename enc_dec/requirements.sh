#!/bin/bash

#1 Installing FairSeq Environment

cd /home/$USER
git clone https://github.com/pytorch/fairseq
python3 -m pip install --editable ./

#2 Python Dependencies

## nltk
python3 -m pip install nltk
python3 src/req.py

## spacy
python3 -m pip install -U spacy
python3 -m spacy download en_core_web_sm

## Levenshtein
python3 -m pip install python-Levenshtein

echo ">>> requirements installed successfully"