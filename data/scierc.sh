#!/bin/bash

mkdir -p SciERC/tmp

python transfer.py transfer $1/train.json SciERC/tmp/train.json [COMPARE,CONJUNCTION]
python transfer.py transfer $1/dev.json SciERC/tmp/dev.json [COMPARE,CONJUNCTION]
python transfer.py transfer $1/test.json SciERC/tmp/test.json [COMPARE,CONJUNCTION]

python process.py process SciERC/tmp/train.json SciERC/train.json allenai/scibert_scivocab_uncased
python process.py process SciERC/tmp/dev.json SciERC/dev.json allenai/scibert_scivocab_uncased
python process.py process SciERC/tmp/test.json SciERC/test.json allenai/scibert_scivocab_uncased

rm -rf SciERC/tmp