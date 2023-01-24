library(tidyverse)
download_path = "C:/Users/grace/Downloads/results2"
data = read_tsv(download_path)

data = mutate(data, Category = str_replace(Category, 'Pretrained-Bio', 'Pretrained-Science'))
data = mutate(data, Model_Type = str_replace(Model_Type, 'pretrained_', ""))
data = mutate(data, Model_Type = str_replace(Model_Type, 'wiki_subword', "1"))
data = mutate(data, Model_Type = str_replace(Model_Type, 'crawl_subword', "3"))
data = mutate(data, Model_Type = str_replace(Model_Type, 'wiki', "2"))
data = mutate(data, Model_Type = str_replace(Model_Type, 'crawl', '4'))
data = mutate(data, Model_Type = factor(Model_Type, levels = c("Bert", "fasttext_1", "fasttext_2", "fasttext_3", "fasttext_4", "gpt2", "Spacy", "MPNet", "Roberta", "T5", "BioBert", 'BiomedRoberta', "bioWordVec", "SciBert", "SciSpacy","fasttext__cbow", "fasttext__skipgram", "GEOBert")))
data = filter(data, Category != 'Unsure')

#One where we multiplication rate 1 with GEO trained models
gdata = filter(data1, Category == "Geo-Trained" & Multiplication_Rate == 1)
g <- ggplot(data=gdata, aes(x=Model_Type, y=Accuracy)) +
  geom_boxplot() + 
  theme_bw()+
  theme(axis.text.x = element_text(angle=315, vjust=0, hjust=0))+
  labs(x = "Model Type", y = "Retrieval Score")+
  scale_y_continuous(limits = c(0,1))

print(g)
