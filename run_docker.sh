#! /bin/bash

echo "Building the Docker container"
docker build -t srp33/geo_nlp .

#mkdir -p Data/tmp
#mkdir -p Queries
#mkdir -p Assignments
#mkdir -p Manual_Searches
#mkdir -p Similarities Similarities_Chunks Similarities_NonGemma Similarities_nolower
#mkdir -p Models
#mkdir -p Results

#docker run --rm -i -t \
docker run --rm \
    --gpus all \
    -v $(pwd)/Data:/Data \
    -v $(pwd)/Queries:/Queries \
    -v $(pwd)/Assignments:/Assignments \
    -v $(pwd)/Similarities:/Similarities \
    -v $(pwd)/Similarities_Chunks:/Similarities_Chunks \
    -v $(pwd)/Similarities_NonGemma:/Similarities_NonGemma \
    -v $(pwd)/Similarities_nolower:/Similarities_nolower \
    -v $(pwd)/Manual_Searches:/Manual_Searches \
    -v $(pwd)/Models:/Models \
    -v $(pwd)/Results:/Results \
    -v /tmp:/tmp \
    --user $(id -u):$(id -g) \
    srp33/geo_nlp \
    /exec_analysis.sh
