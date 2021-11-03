#!/bin/bash

FILE=/Data/model.bin

if [ ! -f "$FILE" ]; then
    wget -O "$FILE" https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
fi
