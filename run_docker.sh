#! /bin/bash

docker build -t srp33/geo_nlp .

mkdir -p Data/tmp
mkdir -p Queries
mkdir -p Assignments
mkdir -p Manual_Searches
mkdir -p Similarities
mkdir -p Models
mkdir -p Results

#docker run --rm -d \
docker run --rm -i -t \
    --gpus all \
    -v $(pwd)/Data:/Data \
    -v $(pwd)/Queries:/Queries \
    -v $(pwd)/Assignments:/Assignments \
    -v $(pwd)/Similarities:/Similarities \
    -v $(pwd)/Manual_Searches:/Manual_Searches \
    -v $(pwd)/Models:/Models \
    -v $(pwd)/Results:/Results \
    -v /tmp:/tmp \
    --user $(id -u):$(id -g) \
    srp33/geo_nlp \
    /exec_analysis.sh
