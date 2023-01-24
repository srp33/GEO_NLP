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

#Line graph with multiplication rates with best, worst, and average
averaged <- group_by(data, Multiplication_Rate) %>%
  summarize(Avg_Accuracy = mean(Accuracy))
print(averaged)
cbPalette = c("#56B4E9", "#E69F00", "#CC79A7", "#009E73", "#F0E442", "#0072B2", "#D55E00")
multiplication = c(log(1), log(2), log(5), log(10), log(100), log(300))
showing = filter(data, Model_Type == 'MPNet'|Model_Type == 'Roberta'|Model_Type == 'BioBert')
#showing = mutate(showing, Multiplication_Rate = log(Multiplication_Rate))
l <- ggplot(averaged, aes(x=log(Multiplication_Rate), y=Avg_Accuracy, color='Average')) +
  #geom_point(data=data, aes(x=Multiplication_Rate, y=Accuracy, color=Model_Type))+
  #geom_jitter(data=data, aes(x=Multiplication_Rate, y=Accuracy, color=Model_Type))+
  geom_smooth(formula = y ~ x, method = "lm", se=FALSE) +
  geom_smooth(data=showing, aes(x=log(Multiplication_Rate), y=Accuracy, color=Model_Type), formula = y ~ x, method = "lm", se=FALSE) +
  geom_point(data=showing, aes(x=log(Multiplication_Rate), y=Accuracy, color=Model_Type))+
  geom_jitter(data=showing, aes(x=log(Multiplication_Rate), y=Accuracy, color=Model_Type))+
  theme_bw()+
  labs(x = "Imbalance (Log Scale)", y = 'Retrieval Score')+
  geom_vline(xintercept= multiplication, color = "#000000", linetype = "longdash", show.legend=TRUE)+
  scale_colour_manual(name = 'Models', values=c(Average="#56B4E9", BioBert = "#E69F00", MPNet =  "#CC79A7", Roberta = "#009E73"))+
  scale_x_continuous(breaks= c(log(1), log(2), log(5), log(10), log(100), log(300)), labels = c("1", "2", "5", "10", "100", "300"))
#ggsave("C:/Users/grace/Downloads/imbalances.png", plot = l, width=20, height=12, units='cm')

print(l)
