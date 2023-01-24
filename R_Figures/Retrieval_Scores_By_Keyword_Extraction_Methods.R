library(tidyverse)
data = read_tsv("C:/Users/grace/Downloads/keyword_results.tsv")
data = mutate(data, Multiplication_Rate = factor(Multiplication_Rate))
data = mutate(data, Num_Keywords = factor(Num_Keywords, levels = c(2, 4, 8, 16, 32, 'full_text')))
k <- ggplot(data, aes(x = Num_Keywords, y = Accuracy)) + 
  geom_col()+
  facet_wrap(~Keyword_Extractor) + 
  theme_bw() + 
  labs(y = 'Retrieval Score')
print(k)
