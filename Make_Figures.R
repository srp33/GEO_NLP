#install.packages(c("knitr", "tidyverse", "writexl"))

library(knitr)
library(tidyverse)
library(writexl)

########################################################################
# Read data in and preprocess it.
########################################################################

make_query_factor = function(data) {
  data %>%
    mutate(Query = factor(Query, levels = c("juvenile_idiopathic_arthritis", "triple_negative_breast_carcinoma", "down_syndrome", "bipolar_disorder", "parkinson_disease", "neuroblastoma"))) %>%
    mutate(Query = fct_recode(Query,
                              "Triple negative breast carcinoma (n = 24)" = "triple_negative_breast_carcinoma",
                              "Parkinson's disease (n = 109)" = "parkinson_disease",
                              "Neuroblastoma (n = 121)" = "neuroblastoma",
                              "Juvenile idiopathic arthritis (n = 12)" = "juvenile_idiopathic_arthritis",
                              "Down syndrome (n = 30)" = "down_syndrome",
                              "Bipolar disorder (n = 34)" = "bipolar_disorder")) %>%
    return()
}

make_query_factor_2 = function(data) {
  data %>%
    mutate(Query = fct_recode(Query,
                            "Triple negative breast carcinoma" = "triple_negative_breast_carcinoma",
                            "Parkinson's disease" = "parkinson_disease",
                            "Neuroblastoma" = "neuroblastoma",
                            "Juvenile idiopathic arthritis" = "juvenile_idiopathic_arthritis",
                            "Down syndrome" = "down_syndrome",
                            "Bipolar disorder" = "bipolar_disorder")) %>%
    return()
}

manual_results = read_tsv("Results/Manual_Searches.tsv.gz") %>%
  make_query_factor() %>%
  mutate(Metric = factor(Metric, levels=c("Precision", "Recall", "F1 score")))

results = read_tsv("Results/Metrics.tsv.gz") %>%
  make_query_factor() %>%
  mutate(Multiplication_Rate = factor(Multiplication_Rate, levels = c("1", "2", "5", "10", "100", "300", "all"))) %>%
  mutate(Method = factor(Method, levels = sort(unique(Method))))

results_chunks = read_tsv("Results/Metrics_Chunks.tsv.gz") %>%
  make_query_factor() %>%
  mutate(Multiplication_Rate = factor(Multiplication_Rate, levels = c("1", "2", "5", "10", "100", "300", "all"))) %>%
  mutate(Method = factor(Method, levels = sort(unique(Method))))

embedding_sizes = read_tsv("Results/Embedding_Sizes.tsv.gz")
checkpoint_metadata = read_tsv("Results/Checkpoint_Metadata.tsv.gz")
textlength_vs_distance = read_tsv("Results/TextLength_vs_Distance.tsv.gz") %>%
  make_query_factor_2()

top_gemma_candidates = read_tsv("Results/Top_Gemma_Candidates.tsv.gz")
top_nongemma_candidates = read_tsv("Results/Top_NonGemma_Candidates.tsv.gz")

########################################################################
# Create graphs for manual searches.
########################################################################

dir.create("Figures", showWarnings = FALSE)

plot_data = mutate(manual_results, Top_Num = as.integer(Top_Num)) %>%
  mutate(Search_Type = factor(Search_Type, levels = c("Tag ontology term", "Tag ontology term plus synonyms", "MeSH term")))

ggplot(plot_data, aes(x = Top_Num, y = Value, color = Metric, linetype=Search_Type)) +
  geom_line() +
  geom_point() +
  facet_wrap(vars(Query), scales="free_x", ncol = 2) +
  xlab("Number of returned series") +
  ylab("Metric value") +
  theme_bw(base_size = 17) +
  scale_color_manual(values = c('Precision' = '#1b9e77', 'Recall' = '#7570b3', 'F1 score' = '#d95f02')) +
  scale_linetype_manual(values = c("MeSH term" = "solid", "Tag ontology term" = "dashed", "Tag ontology term plus synonyms" = "dotdash"), name = "Search type")

ggsave("Figures/Manual_Searches.pdf", width=12.5, height=8.5, unit="in")

########################################################################
# Compare performance with or without text chunking.
########################################################################

diff_data = inner_join(results, results_chunks, by = c("Query", "Method", "Multiplication_Rate", "Top_Num", "Metric")) %>%
  mutate(Difference = Value.y - Value.x) %>%
  # select(-Value.x, -Value.y) %>%
  filter(Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC")

pull(diff_data, Difference) %>%
  median(na.rm = TRUE) %>%
  print()

group_by(diff_data, Method) %>%
  summarize(Difference = median(Difference, na.rm=TRUE)) %>%
  arrange(Method) %>%
  kable(format="simple") %>%
  write("Tables/Chunking_Improvement.md")

########################################################################
# Create graphs for model-based approaches.
########################################################################

set.seed(1)

filter(results, Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC") %>%
  ggplot(aes(x = Value, y = fct_rev(Query))) +
  geom_jitter(alpha = 0.2) +
  geom_boxplot(alpha = 0.0, outlier.shape = NA) +
  xlab("Area under precision-recall curve") +
  ylab("") +
  theme_bw(base_size = 14) +
  theme(plot.margin = margin(l = 0, t = 3, r = 5, b = 3, unit = "mm"))

ggsave("Figures/AUPRC_by_Query.pdf", width=6.5, height=3.5, unit="in")

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

ggsave("Figures/AUPRC_by_Checkpoint.pdf", width=6.5, height=9, unit="in")

unique_embedding_sizes = pull(embedding_sizes, Embedding_Size) %>%
  unique() %>%
  sort()

filter(results, Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC") %>%
  group_by(Query) %>%
  reframe(Method = Method, AUPRC_Rank = rank(-Value, ties.method = "min")) %>%
  group_by(Method) %>%
  summarize(Median_AUPRC_Rank = median(AUPRC_Rank)) %>%
  dplyr::rename(Checkpoint = Method) %>%
  left_join(embedding_sizes) %>%
  left_join(checkpoint_metadata) %>%
  mutate(Has_Embedding = !is.na(Embedding_Size)) %>%
  mutate(Embedding_Size = ifelse(is.na(Embedding_Size), 400, Embedding_Size)) %>%
  mutate(Data_Source_Type = ifelse(is.na(Data_Source_Type), "None", Data_Source_Type)) %>%
  mutate(Data_Source_Type = factor(Data_Source_Type, levels = c("General", "Scientific", "Biomedical", "None"))) %>%
  ggplot(aes(x = Median_AUPRC_Rank,
             y = fct_reorder(Checkpoint, Median_AUPRC_Rank, .desc=TRUE),
             size = Embedding_Size,
             shape = Has_Embedding,
             color = Data_Source_Type)) +
  geom_point() +
  xlab("Median rank (lower is better)") +
  ylab("") +
  theme_bw(base_size = 16) +
  theme(plot.margin = margin(t = 3, r = 12, unit = "mm")) +
  scale_size_continuous(
    breaks = unique_embedding_sizes,
    name = "Embedding size"
  ) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 4)) +
  scale_color_manual(values = c("General" = "#f46d43", "Scientific" = "#abd9e9", "Biomedical" = "#4575b4", "None" = "black"), name = "Data source type") +
  guides(shape = "none")

ggsave("Figures/AUPRC_Rank_by_Checkpoint.pdf", width=11, height=9, unit="in")

inner_join(embedding_sizes, checkpoint_metadata) %>%
  arrange(Checkpoint) %>%
  kable(format="simple") %>%
  write("Tables/Checkpoint_Metadata.md")

top_three = c("thenlper/gte-large", "sentence-transformers/all-roberta-large-v1", "sentence-transformers/all-mpnet-base-v2")

filter(results, Metric == "AUPRC") %>%
  mutate(Method_Color = if_else(Method == top_three[1], "#d73027", if_else(Method == top_three[2], "#f46d43", if_else(Method == top_three[3], "#fdae61", "black")))) %>%
  mutate(Method_Line_Size = if_else(Method %in% top_three, 1, 0.5)) %>%
  mutate(Method_Alpha = if_else(Method %in% top_three, 1, 0.5)) %>%
  ggplot(aes(x = Multiplication_Rate, y = Value, group = Method, color = I(Method_Color), linewidth = I(Method_Line_Size), alpha = I(Method_Alpha))) +
  geom_line() +
  geom_point() +
  facet_wrap(vars(Query)) +
  xlab("Imbalance ratio") +
  ylab("AUPRC") +
  theme_bw(base_size = 14)

ggsave("Figures/AUPRC_by_Multiplication_Rate.pdf", width=11, height=9, unit="in")

filter(results, Metric == "Recall") %>%
  filter(Multiplication_Rate == "all") %>%
  dplyr::rename(Recall = Value) %>%
  mutate(Top_Num = str_replace_all(Top_Num, "Recall_Top_", "")) %>%
  mutate(Top_Num = as.integer(Top_Num)) %>%
  mutate(Top_Num = factor(Top_Num)) %>%
  mutate(Method_Color = if_else(Method == top_three[1], "#d73027", if_else(Method == top_three[2], "#f46d43", if_else(Method == top_three[3], "#fdae61", "black")))) %>%
  mutate(Method_Line_Size = if_else(Method %in% top_three, 1, 0.5)) %>%
  mutate(Method_Alpha = if_else(Method %in% top_three, 1, 0.5)) %>%
  ggplot(aes(x = Top_Num, y = Recall, group = Method, color = I(Method_Color), linewidth = I(Method_Line_Size), alpha = I(Method_Alpha))) +
  geom_line() +
  geom_point() +
  facet_wrap(vars(Query)) +
  xlab("Number of top-ranked series") +
  ylab("Recall (sensitivity)") +
  theme_bw(base_size = 14)

ggsave("Figures/Recall_by_Top_Num.pdf", width=11, height=9, unit="in")

select(top_gemma_candidates, -Score) %>%
  mutate(Model = str_replace_all(Model, "____", "/")) %>%
  make_query_factor_2() %>%
  group_by(Query, Series, Series_Title, Series_Summary, Series_Overall_Design) %>%
  summarize(`Model Count` = n()) %>%
  ungroup() %>%
  arrange(desc(`Model Count`)) %>%
  head(n = 25) %>%
  dplyr::rename(`Series title` = Series_Title) %>%
  dplyr::rename(`Series summary` = Series_Summary) %>%
  dplyr::rename(`Series overall design` = Series_Overall_Design) %>%
  mutate(`Current Gemma tag(s)` = "") %>%
  mutate(`Comment(s)` = "") %>%
  mutate(`Strong case` =  "No") -> top_gemma_candidates_table

# I added Current Gemma tag(s) to the spreadsheet on May 1, 2024.

# FYI: Uncomment this only if you want to overwrite the XLSX file.
# write_xlsx(top_gemma_candidates_table, "Tables/Top_Gemma_Candidates.xlsx")

top_gemma_candidates_excel = readxl::read_xlsx("Tables/Top_Gemma_Candidates.xlsx")

pull(top_gemma_candidates_excel, `Strong case`) %>%
  factor(levels = c("Yes", "No", "Maybe")) %>%
  table() %>%
  print()
#   Yes    No Maybe 
#     9    13     3 

kable(top_gemma_candidates_excel, format="simple") %>%
  write("Tables/Top_Gemma_Candidates.md")

mutate(top_nongemma_candidates, Model = str_replace_all(Model, "____", "/")) %>%
  mutate(Query = fct_recode(Query,
                            "Triple negative breast carcinoma" = "triple_negative_breast_carcinoma",
                            "Parkinson's disease" = "parkinson_disease",
                            "Neuroblastoma" = "neuroblastoma",
                            "Juvenile idiopathic arthritis" = "juvenile_idiopathic_arthritis",
                            "Down syndrome" = "down_syndrome",
                            "Bipolar disorder" = "bipolar_disorder")) %>%
  dplyr::rename(`Series title` = Series_Title) %>%
  dplyr::rename(`Series summary` = Series_Summary) %>%
  dplyr::rename(`Series overall design` = Series_Overall_Design) %>%
  mutate(`Comment(s)` = "") %>%
  mutate(`Relevant to medical condition` =  "No") %>%
  mutate(`Relevant to related condition(s)` = "No") %>%
  mutate(`Primary sample(s)` = "No") %>%
  mutate(`Cell line(s)` = "No") %>%
  mutate(`Xenograft(s)` = "No") -> top_nongemma_candidates_table

textlength_vs_distance_rho = group_by(textlength_vs_distance, Query) %>%
  summarize(spearman_rho = cor(TextLength, Distance, method="spearman"), .groups = "drop")

textlength_vs_distance <- left_join(textlength_vs_distance, textlength_vs_distance_rho, by = "Query")

ggplot(textlength_vs_distance, aes(x = TextLength, y = Distance)) +
  geom_point() +
  facet_wrap(vars(Query)) +
  geom_text(data = textlength_vs_distance_rho, aes(label = sprintf("rho = %.2f", spearman_rho), x = Inf, y = Inf), 
            hjust = 1.4, vjust = 4, check_overlap = TRUE, size=4.5) +
  xlab("Text length (# of characters)") +
  theme_bw(base_size = 16)

ggsave("Figures/Gemma_TextLength_vs_Distance.pdf", width=11, height=9, unit="in")

# FYI: Uncomment this only if you want to overwrite the XLSX file.
#write_xlsx(top_nongemma_candidates_table, "Tables/Top_NonGemma_Candidates.xlsx")

top_nongemma_candidates_excel = readxl::read_xlsx("Tables/Top_NonGemma_Candidates.xlsx") %>%
  mutate(`Relevant to medical condition` = factor(`Relevant to medical condition`, levels=c("Yes", "No", "Maybe"))) %>%
  mutate(Query = factor(Query, levels = c("Triple negative breast carcinoma", "Parkinson's disease", "Neuroblastoma", "Juvenile idiopathic arthritis", "Down syndrome", "Bipolar disorder")))

pull(top_nongemma_candidates_excel, `Relevant to medical condition`) %>%
  table() %>%
  print()
# Maybe    No   Yes 
# 17       82   201

group_by(top_nongemma_candidates_excel, Query, `Relevant to medical condition`) %>%
  summarize(Count = n()) %>%
  ggplot(aes(x = Query, y = Count, fill = `Relevant to medical condition`)) +
  geom_col() +
  coord_flip() +
  xlab("") +
  scale_fill_brewer(palette = "Set2") +
  theme_bw(base_size=16)

ggsave("Figures/NonGemma_Relevance.pdf", width=11, height=9, unit="in")

filter(top_nongemma_candidates_excel, `Relevant to medical condition` == "No") %>%
  pull(`Relevant to related condition(s)`) %>%
  table() %>%
  print()
# Maybe    No   Yes 
# 8        24    50

filter(top_nongemma_candidates_excel, `Relevant to medical condition` == "Yes") %>%
  pivot_longer(cols=c(`Primary sample(s)`, `Cell line(s)`, `Xenograft(s)`), names_to = "Sample type", values_to = "Response") %>%
  filter(Response == "Yes") %>%
  mutate(`Sample type` = factor(`Sample type`, levels = c("Primary sample(s)", "Cell line(s)", "Xenograft(s)"))) %>%
  group_by(Query, `Sample type`) %>%
  summarize(Count = n()) %>%
  ggplot(aes(x = Query, y = Count, fill = `Sample type`)) +
  geom_col() +
  coord_flip() +
  xlab("") +
  scale_fill_brewer(palette = "Set2") +
  theme_bw(base_size=16)

ggsave("Figures/NonGemma_SampleType.pdf", width=11, height=9, unit="in")

# TODO:
# Send Anna the JSON file with the gte embeddings for all of GEO (make sure it has all).
# Send Anna the TSV file with other metadata.