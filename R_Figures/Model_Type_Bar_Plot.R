#all models bar plot

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

data2 = group_by(data1, Model_Type) %>% summarize(`Retrieval Score`= mean(Accuracy))
spec  = filter(data2, Model_Type == "Roberta" | Model_Type == "MPNet" | Model_Type == "T5")
g <- ggplot(data=data2, aes(x=Model_Type, y=`Retrieval Score`)) +
  geom_col(fill='light grey', color = 'black')+
  geom_col(data=spec, aes(x=Model_Type, y=`Retrieval Score`), fill='dark grey', color='black', lwd = 1)+
  geom_point(data = data1, aes(x=Model_Type,y=Accuracy, color=Category))+
  geom_jitter(data = data1, aes(x=Model_Type,y=Accuracy, color=Category))+
  scale_colour_manual(name = 'Category', values=c("#56B4E9","#CC79A7","#E69F00"))+ 
  theme_bw()+
  labs(x = "Model Type", y = "Retrieval Score")+
  theme(axis.text.x = element_text(angle=315, vjust=0, hjust=0))+
  scale_y_continuous(limits = c(0,1))
#ggsave(f"{download_path}/allbarplot.png", plot = g, width=20, height=12, units='cm')
print(g)
