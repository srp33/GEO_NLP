library(tidyverse)
data = read_tsv("C:/Users/grace/Downloads/length_retrieval.tsv")
#spearman coefficient & log transformation
print(cor.test(data$Length,data$Position, method ='spearman'))
log_data = mutate(data, Length=log(Length)) %>% mutate(Position=log(Position)) %>% mutate(Threshold=log(Threshold))


data1=filter(data, Query=='q1')
l1<-ggplot(data6, aes(x = Length, y=Position)) +
  geom_point()+
  geom_jitter()+
  geom_smooth(method='lm')+
  facet_wrap(~Query)+
  geom_hline(yintercept= data1$Threshold)+
  labs(x='Length(log transformed)', y='Position (log transformed)')
  print(l1)
data2=filter(data, Query=='q2')
l2<-ggplot(data2, aes(x = Length, y=Position)) +
  geom_point()+
  geom_jitter()+
  geom_smooth(method='lm')+
  facet_wrap(~Query)+
  geom_hline(yintercept= data2$Threshold)+
  labs(x='Length(log transformed)', y='Position (log transformed)')
  print(l2)
data3=filter(data, Query=='q3')
l3<-ggplot(data3, aes(x = Length, y=Position)) +
  geom_point()+
  geom_jitter()+
  geom_smooth(method='lm')+
  facet_wrap(~Query)+
  geom_hline(yintercept= data3$Threshold)+
  labs(x='Length(log transformed)', y='Position (log transformed)')
  print(l3)
data4=filter(data, Query=='q4')
l4<-ggplot(data4, aes(x = Length, y=Position)) +
  geom_point()+
  geom_jitter()+
  geom_smooth(method='lm')+
  facet_wrap(~Query)+
  geom_hline(yintercept= data4$Threshold)+
  labs(x='Length(log transformed)', y='Position (log transformed)')
  print(l4)
data5=filter(data, Query=='q5')
l5<-ggplot(data5, aes(x = Length, y=Position)) +
  geom_point()+
  geom_jitter()+
  geom_smooth(method='lm')+
  facet_wrap(~Query)+
  geom_hline(yintercept= data5$Threshold)+
  labs(x='Length(log transformed)', y='Position (log transformed)')
  print(l5)
data6=filter(data, Query=='q6')
l6<-ggplot(data6, aes(x = Length, y=Position)) +
  geom_point()+
  geom_jitter()+
  geom_smooth(method='lm')+
  facet_wrap(~Query)+
  geom_hline(yintercept= data6$Threshold)+
  labs(x='Length(log transformed)', y='Position (log transformed)')
print(l6)
