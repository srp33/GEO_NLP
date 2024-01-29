#! /bin/bash

docker build -t srp33/geo_nlp .

mkdir -p Data
mkdir -p Models Models/custom
mkdir -p Queries
mkdir -p Results

#docker run --rm -d \
docker run --rm -i -t \
    -v $(pwd)/Data/:/Data/ \
    -v $(pwd)/Models:/Models \
    -v $(pwd)/Queries:/Queries \
    -v $(pwd)/Results:/Results \
    -v /tmp:/tmp \
    --user $(id -u):$(id -g) \
    srp33/geo_nlp \
    /exec_analysis.sh
