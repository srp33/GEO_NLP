docker build -t srp33/data_master .

mkdir -p Data/q1 Data/q2 Data/q3 Data/q4 Data/q5 Data/q6
mkdir -p Models Models/custom
mkdir -p Results

#docker run --rm -d \
docker run --rm -i -t \
    -v $(pwd)/Data/:/Data/ \
    -v $(pwd)/Models:/Models \
    -v $(pwd)/Results:/Results \
    -v /tmp:/tmp \
    --user $(id -u):$(id -g) \
    srp33/data_master \
    /exec_analysis.sh
