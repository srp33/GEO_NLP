This repository contains data and code from our evaluation of using language models to find datasets in Gene Expression Omnibus.

You will need to create a directory called `Models` and store your OpenAI API key in a file called `Models/open_ai.key`.

To execute the analysis on a Linux system, first install the [Docker Desktop software](https://docs.docker.com/desktop/setup/install/linux/).
Then execute the `run_docker.sh` bash script at the command line.x
This script should build a Docker container and execute the analysis.
It will take many hours (or days, depending on your computing power) to execute the analysis, even with GPUs enabled.

You can then run the `Make_Figures.R` script to create the figures and tables. This is easier in RStudio, rather than in Docker. Uncomment and execute the first line of code to install the necessary packages first.
