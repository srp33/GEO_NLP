#!/bin/bash

FILE="/Data/wiki.en.vec"

if [ ! -f "$FILE" ]
then
  wget -O "$FILE" https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
fi
