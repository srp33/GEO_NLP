#install.packages(c("knitr", "tidyverse", "writexl"))

library(knitr)
library(tidyverse)
library(writexl)

results = read_tsv("../Results/Metrics.tsv.gz") %>%
  mutate(Query = factor(Query, levels = c("triple_negative_breast_carcinoma", "juvenile_idiopathic_arthritis", "down_syndrome", "bipolar_disorder", "neuroblastoma", "parkinson_disease"))) %>%
  mutate(Query = fct_recode(Query,
                            "Triple negative breast carcinoma (n = 11)" = "triple_negative_breast_carcinoma",
                            "Parkinson's disease (n = 112)" = "parkinson_disease",
                            "Neuroblastoma (n = 93)" = "neuroblastoma",
                            "Juvenile idiopathic arthritis (n = 13)" = "juvenile_idiopathic_arthritis",
                            "Down syndrome (n = 27)" = "down_syndrome",
                            "Bipolar disorder (n = 34)" = "bipolar_disorder")) %>%
  mutate(Multiplication_Rate = factor(Multiplication_Rate, levels = c("1", "2", "5", "10", "100", "300", "all"))) %>%
  mutate(Method = factor(Method, levels = sort(unique(Method))))

embedding_sizes = read_tsv("../Results/Embedding_Sizes.tsv.gz")

top_other_candidates = read_tsv("../Results/Top_Other_Candidates.tsv.gz")

filter(results, Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC") %>%
  ggplot(aes(x = Value, y = fct_rev(Query))) +
  geom_jitter(alpha = 0.2) +
  geom_boxplot(alpha = 0.0, outlier.shape = NA) +
  xlab("Area under precision-recall curve") +
  ylab("") +
  theme_bw(base_size = 14) +
  theme(plot.margin = margin(l = 0, t = 3, r = 5, b = 3, unit = "mm"))

dir.create("../Figures", showWarnings = FALSE)

ggsave("../Figures/AUPRC_by_Query.pdf", width=6.5, height=3.5, unit="in")

filter(results, Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC") %>%
  ggplot(aes(x = Value, y = fct_rev(Method))) +
  geom_point(aes(color = Query, shape = Query), size = 2) +
  geom_boxplot(outlier.shape = NA, alpha = 0) +
  xlab("Area under precision-recall curve") +
  ylab("") +
  theme_bw() +
  theme(legend.position = "bottom") +
  theme(legend.title = element_blank()) +
  theme(plot.margin = margin(t = 3, r = 12, unit = "mm")) +
  guides(color = guide_legend(ncol=1))
#  scale_color_manual(values=c("factor_level1"="red", "factor_level2"="blue"))

ggsave("../Figures/AUPRC_by_Checkpoint.pdf", width=6.5, height=9, unit="in")

filter(results, Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC") %>%
  group_by(Query) %>%
  reframe(Method = Method, AUPRC_Rank = rank(-Value, ties.method = "min")) %>%
  group_by(Method) %>%
  summarize(Median_AUPRC_Rank = median(AUPRC_Rank)) %>%
  ggplot(aes(x = Median_AUPRC_Rank, y = fct_reorder(Method, Median_AUPRC_Rank, .desc=TRUE))) +
  geom_col() +
  xlab("Median rank (lower is better)") +
  ylab("") +
  theme_bw() +
  theme(legend.position = "bottom") +
  theme(legend.title = element_blank()) +
  theme(plot.margin = margin(t = 3, r = 12, unit = "mm")) +
  guides(color = guide_legend(ncol=3))

ggsave("../Figures/AUPRC_Rank_by_Checkpoint.pdf", width=11, height=9, unit="in")

filter(results, Metric == "AUPRC") %>%
  mutate(Method_Color = if_else(Method == "sentence-transformers/all-roberta-large-v1", "#d73027", if_else(Method == "thenlper/gte-large", "#f46d43", if_else(Method == "sentence-transformers/paraphrase-TinyBERT-L6-v2", "#fdae61", "black")))) %>%
  mutate(Method_Line_Size = if_else(Method %in% c("sentence-transformers/all-roberta-large-v1", "thenlper/gte-large", "sentence-transformers/paraphrase-TinyBERT-L6-v2"), 1, 0.5)) %>%
  mutate(Method_Alpha = if_else(Method %in% c("sentence-transformers/all-roberta-large-v1", "thenlper/gte-large", "sentence-transformers/paraphrase-TinyBERT-L6-v2"), 1, 0.5)) %>%
  ggplot(aes(x = Multiplication_Rate, y = Value, group = Method, color = I(Method_Color), size = I(Method_Line_Size), alpha = I(Method_Alpha))) +
  geom_line() +
  geom_point() +
  facet_wrap(vars(Query)) +
  xlab("Imbalance ratio") +
  ylab("AUPRC") +
  theme_bw(base_size = 14)

ggsave("../Figures/AUPRC_by_Multiplication_Rate.pdf", width=11, height=9, unit="in")

filter(results, Metric != "AUPRC") %>%
  filter(Multiplication_Rate == "all") %>%
  dplyr::rename(Num_Top = Metric) %>%
  dplyr::rename(Recall = Value) %>%
  mutate(Num_Top = str_replace_all(Num_Top, "Recall_Top_", "")) %>%
  mutate(Num_Top = as.integer(Num_Top)) %>%
  mutate(Num_Top = factor(Num_Top)) %>%
  mutate(Method_Color = if_else(Method == "sentence-transformers/all-roberta-large-v1", "#d73027", if_else(Method == "thenlper/gte-large", "#f46d43", if_else(Method == "sentence-transformers/paraphrase-TinyBERT-L6-v2", "#fdae61", "black")))) %>%
  mutate(Method_Line_Size = if_else(Method %in% c("sentence-transformers/all-roberta-large-v1", "thenlper/gte-large", "sentence-transformers/paraphrase-TinyBERT-L6-v2"), 1, 0.5)) %>%
  mutate(Method_Alpha = if_else(Method %in% c("sentence-transformers/all-roberta-large-v1", "thenlper/gte-large", "sentence-transformers/paraphrase-TinyBERT-L6-v2"), 1, 0.5)) %>%
  ggplot(aes(x = Num_Top, y = Recall, group = Method, color = I(Method_Color), size = I(Method_Line_Size), alpha = I(Method_Alpha))) +
  geom_line() +
  geom_point() +
  facet_wrap(vars(Query)) +
  xlab("Number of top-ranked series") +
  ylab("Recall (sensitivity)") +
  theme_bw(base_size = 14)

#FYI: This will only show Num_Top values when the number of Testing samples is greater than or equal to the threshold.

ggsave("../Figures/Recall_by_Num_Top.pdf", width=11, height=9, unit="in")

select(top_other_candidates, -Score) %>%
  mutate(Model = str_replace_all(Model, "____", "/")) %>%
  mutate(Query = fct_recode(Query,
                            "Triple negative breast carcinoma" = "triple_negative_breast_carcinoma",
                            "Parkinson's disease" = "parkinson_disease",
                            "Neuroblastoma" = "neuroblastoma",
                            "Juvenile idiopathic arthritis" = "juvenile_idiopathic_arthritis",
                            "Down syndrome" = "down_syndrome",
                            "Bipolar disorder" = "bipolar_disorder")) %>%
  group_by(Query, Series) %>%
  summarize(`Model Count` = n()) %>%
  ungroup() %>%
  arrange(desc(`Model Count`)) %>%
  head(n = 25) %>%
  mutate(`GEO title` = "") %>%
  mutate(`GEO summary` = "") %>%
  mutate(`GEO overall design` = "") %>%
  mutate(`Current Gemma tag(s)` = "") %>%
  mutate(`Comment(s)` = "") %>%
  mutate(`Strong case` =  "No") -> top_candidates_table

# I added Current Gemma tag(s) to the spreadsheet on March 13-15, 2024.

# FYI: Uncomment this only if you want to overwrite the XLSX file.
#write_xlsx(top_candidates_table, "../Tables/Top_Candidates.xlsx")

top_candidates_excel = readxl::read_xlsx("../Tables/Top_Candidates.xlsx")

pull(top_candidates_excel, `Strong case`) %>%
  factor(levels = c("Yes", "No", "Maybe")) %>%
  table() %>%
  print()

kable(top_candidates_excel, format="simple") %>%
  write("../Tables/Top_Candidates.md")

# TODO:
# Do manual queries in GEO.
#   1. Keyword search using primary term only.
#   2. Keyword search using primary term + synonyms in ontology.
#   3. Search using on MeSH term.

# Visualize / summarize model types.
#   Finish code on the server to save model types.
#   Add those as colors to the geom_col above. Change to a geom_point?
#   Change the sizes of the points according to embedding_sizes.
# Make predictions for the rest of GEO for the top model and create a 
#   spreadsheet that summarizes those predictions and allows for review.
# Do fine tuning with randomly selected 50% of non-Gemma GEO?
# Average across sentences to see if that helps for some models.
# Checks for bias in favor of shorter or longer summary sections.
# Clean up exec_analysis.