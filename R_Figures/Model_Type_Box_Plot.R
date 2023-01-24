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

data1 = mutate(data, Multiplication_Rate = factor(Multiplication_Rate))
dataspec = filter(data1, Model_Type == "Roberta" | Model_Type == "MPNet" | Model_Type == "T5")
g <- ggplot(data=data1, aes(x=Model_Type, y=Accuracy)) +
  geom_boxplot(lwd=1) + 
  geom_boxplot(data = dataspec, aes(x=Model_Type,y=Accuracy), lwd = 2, fill = 'grey')+
  geom_point(aes(color=Category), alpha = .4)+
  geom_jitter(aes(color=Category), alpha = .4)+
  geom_point(data = dataspec, aes(x=Model_Type,y=Accuracy, color=Category), alpha = 1)+
  geom_jitter(data = dataspec, aes(x=Model_Type,y=Accuracy, color=Category), alpha = 1)+
  scale_colour_manual(name = 'Category', values=c("#CC79A7","#56B4E9","#E69F00" ))+ 
  theme_bw()+
  labs(x = "Model Type", y = "Retrieval Score")+
  theme(axis.text.x = element_text(angle=315, vjust=0, hjust=0))+
  scale_y_continuous(limits = c(0,1))
print(g)
#ggsave(f"{download_path}/allmodels.png", plot = g, width=20, height=12, units='cm')
