#! /bin/bash

docker build -t srp33/geo_nlp .

mkdir -p Data/tmp
mkdir -p Queries
mkdir -p Assignments
mkdir -p Similarities
mkdir -p Models
mkdir -p Metrics

#docker run --rm -d \
docker run --rm -i -t \
    -v $(pwd)/Data:/Data \
    -v $(pwd)/Queries:/Queries \
    -v $(pwd)/Assignments:/Assignments \
    -v $(pwd)/Similarities:/Similarities \
    -v $(pwd)/Models:/Models \
    -v $(pwd)/Metrics:/Metrics \
    -v /tmp:/tmp \
    --user $(id -u):$(id -g) \
    srp33/geo_nlp \
    /exec_analysis.sh
#    --gpus all \
