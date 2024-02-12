#! /bin/bash

docker build -t srp33/geo_nlp .

mkdir -p Data
mkdir -p Queries
mkdir -p Assignments
mkdir -p Similarities
mkdir -p Models Models/custom
mkdir -p Metrics

#docker run --rm -i -t \
docker run --rm -d \
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
