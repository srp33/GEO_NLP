library(tidyverse)
data_tibble = read_tsv("/Results/results.tsv")
print(unique(data_tibble$"Extraction Method"))
