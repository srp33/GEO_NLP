docker build -t srp33/data_master .

mkdir -p Data/q1 Data/q2 Data/q3 Data/q4 Data/q5 Data/q6
mkdir -p Models Models/custom
mkdir -p Results

#docker run --rm -i -t \
docker run --rm -d \
    -v $(pwd)/Data/:/Data/ \
    -v $(pwd)/Models:/Models \
    -v $(pwd)/Results:/Results \
    -v /tmp:/tmp \
    srp33/data_master \
    /exec_analysis.sh

#    /bin/bash
# --user $(id -u):$(id -g) \