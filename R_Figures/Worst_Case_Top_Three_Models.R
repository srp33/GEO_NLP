#These show the position of the last relevant series. 
library(tidyverse)
data = read_tsv("C:/Users/grace/Downloads/all_star.tsv")
#data = filter(data, Model_Type == 'all-roberta-large-v1')
data = filter(data, Model_Type %in% c('all-roberta-large-v1', 'all-mpnet-base-v2', 'sentence-t5-xxl'))
a <- ggplot(data, aes(x = Query, y = Worst_Case/3)) + 
  geom_col()+
  theme_bw() + 
  #facet_wrap(~Model_Type)
  labs(y ="Worst Case")
print(a)
