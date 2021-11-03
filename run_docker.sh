docker build -t srp33/data_master .

mkdir -p Data/q1 Data/q2 Data/q3 Data/q4 Data/q5 Data/q6
mkdir -p Results

docker run --rm -i -t \
    -v $(pwd)/Data/:/Data/ \
    -v $(pwd)/Models:/Models \
    -v $(pwd)/Results:/Results \
    srp33/data_master \
    /runMe.sh
