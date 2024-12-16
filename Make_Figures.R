#install.packages(c("knitr", "tidyverse", "writexl"))

library(broom)
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

medical_condition_levels = c("Juvenile idiopathic arthritis", "Triple negative breast carcinoma", "Down syndrome", "Bipolar disorder", "Parkinson's disease", "Neuroblastoma")

#####################################################################
# Prepare spreadsheet for pilot manual review
#####################################################################
top_nongemma_candidates_pilot = read_tsv("Results/Top_NonGemma_Candidates_Pilot.tsv.gz") %>%
  dplyr::select(-Model, -Query)

for (column_name in medical_condition_levels) {
  top_nongemma_candidates_pilot[[column_name]] = "No"
}
top_nongemma_candidates_pilot[["None"]] = "No"
top_nongemma_candidates_pilot[["Investigate more"]] = "No"
set.seed(0)
top_nongemma_candidates_pilot = sample_n(top_nongemma_candidates_pilot, nrow(top_nongemma_candidates_pilot), replace = FALSE)
write_xlsx(top_nongemma_candidates_pilot, "Tables/Top_NonGemma_Candidates_Pilot.xlsx")
#####################################################################

manual_search_summary = read_tsv("Results/Manual_Search_Gemma_Summary.tsv.gz") %>%
  make_query_factor() %>%
  mutate(Search_Type = factor(Search_Type, levels = c("Tag ontology term", "Tag ontology term plus synonyms", "MeSH term"))) %>%
  mutate(Metric = factor(Metric, levels = c("Precision", "Recall", "F1 score"))) %>%
  mutate(Top_Num = as.integer(Top_Num))

manual_search_items = read_tsv("Results/Manual_Search_Items.tsv.gz") %>%
  make_query_factor_2() %>%
  mutate(Search_Type = factor(Search_Type, levels = c("Tag ontology term", "Tag ontology term plus synonyms", "MeSH term"))) %>%
  dplyr::rename(Series = Series_ID)

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
  make_query_factor_2() %>%
  mutate(Query = factor(Query, levels = medical_condition_levels))

top_gemma_candidates = read_tsv("Results/Top_Gemma_Candidates.tsv.gz")
top_nongemma_candidates = read_tsv("Results/Top_NonGemma_Candidates.tsv.gz")

########################################################################
# Create graphs for manual searches.
########################################################################

dir.create("Figures", showWarnings = FALSE)

manual_search_summary %>%
  filter(Metric == "Recall") %>%
  ggplot(aes(x = Top_Num, y = Value, linetype=Search_Type)) +
    geom_line(alpha = 0.5) +
    geom_point() +
    facet_wrap(vars(Query), ncol = 2) +
    xlab("Number of returned series") +
    ylab("Recall (sensitivity)") +
    theme_bw(base_size = 17) +
    scale_linetype_manual(values = c("MeSH term" = "solid", "Tag ontology term" = "dashed", "Tag ontology term plus synonyms" = "dotdash"), name = "Search type") +
    ylim(0, 1)

ggsave("Figures/Manual_Searches.pdf", width=12.5, height=8.5, unit="in")

########################################################################
# Compare performance with or without text chunking.
########################################################################

diff_data = inner_join(results, results_chunks, by = c("Query", "Method", "Multiplication_Rate", "Top_Num", "Metric")) %>%
  mutate(Difference = Value.y - Value.x) %>%
  # dplyr::select(-Value.x, -Value.y) %>%
  filter(Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC")

pull(diff_data, Difference) %>%
  median(na.rm = TRUE) %>%
  print()

group_by(diff_data, Method) %>%
  dplyr::summarize(Difference = median(Difference, na.rm=TRUE)) %>%
  arrange(Method) %>%
  kable(format="simple") %>%
  write("Tables/Chunking_Improvement.md")

########################################################################
# Create graphs for model-based approaches.
########################################################################

set.seed(0)

plot_data = filter(results, Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC")

# Check for normality
group_by(plot_data, Query) %>%
  dplyr::summarize(shapiro.test(Value)$p.value) %>%
  print() # All p-values < 0.05.

p_value = kruskal.test(Value ~ Query, data = plot_data) %>%
  tidy() %>%
  pull(p.value) %>%
  print() # 8.81088e-06

ggplot(plot_data, aes(x = Value, y = fct_rev(Query))) +
  geom_jitter(alpha = 0.2) +
  geom_boxplot(alpha = 0.0, outlier.shape = NA) +
  xlab("Area under precision-recall curve") +
  ylab("") +
  labs(subtitle = str_c("p < 0.001")) +
  theme_bw(base_size = 14) +
  theme(plot.margin = margin(l = 0, t = 3, r = 5, b = 3, unit = "mm")) +
  theme(plot.subtitle = element_text(hjust = 0.5))

ggsave("Figures/AUPRC_by_Query.pdf", width=6.5, height=3.5, unit="in")

# Check for normality
group_by(plot_data, Method) %>%
  dplyr::summarize(p = shapiro.test(Value)$p.value) %>%
  print(n = Inf) # All p-values except one > 0.05.

p_value = kruskal.test(Value ~ Method, data = plot_data) %>%
  tidy() %>%
  pull(p.value) %>%
  print() # 5.329025e-15

ggplot(plot_data, aes(x = Value, y = fct_rev(Method))) +
  geom_point(aes(color = Query, shape = Query), size = 2) +
  geom_boxplot(outlier.shape = NA, alpha = 0) +
  xlab("Area under precision-recall curve") +
  ylab("") +
  labs(subtitle = str_c("p < 0.001")) +
  theme_bw() +
  theme(legend.position = "bottom") +
  theme(legend.title = element_blank()) +
  theme(plot.margin = margin(t = 3, r = 12, unit = "mm")) +
  theme(plot.subtitle = element_text(hjust = 0.5)) +
  guides(color = guide_legend(ncol=1))

ggsave("Figures/AUPRC_by_Checkpoint.pdf", width=6.5, height=9, unit="in")

unique_embedding_sizes = pull(embedding_sizes, Embedding_Size) %>%
  unique() %>%
  sort()

plot_data = filter(results, Multiplication_Rate == "all") %>%
  filter(Metric == "AUPRC") %>%
  group_by(Query) %>%
  reframe(Method = Method,
          AUPRC = Value,
          AUPRC_Rank = rank(-Value, ties.method = "min")) %>%
  group_by(Method) %>%
  dplyr::summarize(Median_AUPRC = median(AUPRC),
                   Median_AUPRC_Rank = median(AUPRC_Rank)) %>%
  dplyr::rename(Checkpoint = Method) %>%
  left_join(embedding_sizes) %>%
  left_join(checkpoint_metadata) %>%
  mutate(Has_Embedding = !is.na(Embedding_Size)) %>%
  mutate(Embedding_Size = ifelse(is.na(Embedding_Size), 400, Embedding_Size)) %>%
  mutate(Data_Source_Type = ifelse(is.na(Data_Source_Type), "None", Data_Source_Type)) %>%
  mutate(Data_Source_Type = factor(Data_Source_Type, levels = c("General", "Scientific", "Biomedical", "None")))

ggplot(plot_data, aes(x = Median_AUPRC_Rank,
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

plot_data = filter(plot_data, Has_Embedding == TRUE)

# Correlation between rank and embedding size.
cor.test(plot_data$Median_AUPRC, plot_data$Embedding_Size, method="spearman") %>%
  print()
# S = 2216.4, p-value = 0.004253
# alternative hypothesis: true rho is not equal to 0
# sample estimates:
#   rho 
# 0.5069084

stat_data = filter(plot_data, Data_Source_Type %in% c("General", "Biomedical"))
wilcox.test(Median_AUPRC ~ Data_Source_Type, data = stat_data) %>%
  print()
# p-value = 0.4557

inner_join(embedding_sizes, checkpoint_metadata) %>%
  arrange(Checkpoint) %>%
  mutate(Fine_Tuning = ifelse(is.na(Fine_Tuning), "", Fine_Tuning)) %>%
  dplyr::rename(Model = Checkpoint,
                `Embedding size` = Embedding_Size,
                `Data source type` = Data_Source_Type,
                `Model category` = Model_Category,
                `Fine tuning` = Fine_Tuning
                ) %>%
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
  ylim(0, 1) +
  theme_bw(base_size = 14)

ggsave("Figures/Recall_by_Top_Num_All_Checkpoints.pdf", width=11, height=9, unit="in")

########################################################################
# Compare manual vs. model-based approaches.
########################################################################

# AUPRC doesn't make as much sense because there is a variable
# number of results returned from GEO, so it's harder to compare them.
#metric = "AUPRC"

metric = "Recall"

checkpoints = filter(results, Metric == metric) %>%
  filter(Multiplication_Rate == "all") %>%
  filter(Method %in% top_three) %>%
  dplyr::select(Query, Method, Top_Num, Metric, Value) %>%
  mutate(Top_Num = str_replace_all(Top_Num, "Recall_Top_", "")) %>%
  mutate(Top_Num = as.integer(Top_Num)) %>%
  mutate(Top_Num = factor(Top_Num)) %>%
  mutate(Overall_Category = "Language model")

manual = filter(manual_search_summary, Metric == metric) %>%
  dplyr::select(Query, Search_Type, Top_Num, Metric, Value) %>%
  dplyr::rename(Method = Search_Type) %>%
  mutate(Top_Num = as.integer(Top_Num)) %>%
  mutate(Top_Num = factor(Top_Num)) %>%
  mutate(Overall_Category = "Manual GEO search")

bind_rows(checkpoints, manual) %>%
  mutate(Overall_Category = factor(Overall_Category, levels = c("Manual GEO search", "Language model"))) %>%
  ggplot(aes(x = Top_Num, y = Value, group = Method)) +
  geom_line(alpha = 0.6, size = 3, aes(color = Overall_Category)) +
  scale_color_manual(values = c("Manual GEO search" = "#2c7bb6", "Language model" = "#d7191c")) +
  geom_point(aes(shape = Method), size = 2) +
  facet_wrap(vars(Query), ncol = 2) +
  xlab("Number of top-ranked series") +
  ylab(metric) +
  labs(color = "Overall category") +
  ylim(0, 1) +
  theme_bw(base_size = 14)

ggsave("Figures/Compare_Top_Num_Manual_vs_Checkpoints_Gemma.pdf", width=11, height=9, unit="in")

########################################################################
# Summarize top Gemma candidates
########################################################################

dplyr::select(top_gemma_candidates, -Score) %>%
  mutate(Model = str_replace_all(Model, "____", "/")) %>%
  make_query_factor_2() %>%
  group_by(Query, Series, Series_Title, Series_Summary, Series_Overall_Design) %>%
  dplyr::summarize(`Model Count` = n()) %>%
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
  factor(levels = c("Yes", "Maybe", "No")) %>%
  table() %>%
  print()
#   Yes    No Maybe 
#     9    13     3 

kable(top_gemma_candidates_excel, format="simple") %>%
  write("Tables/Top_Gemma_Candidates.md")

textlength_vs_distance_rho = group_by(textlength_vs_distance, Query) %>%
  dplyr::summarize(
    spearman_rho = cor(TextLength, Distance, method="spearman"),
    spearman_p = cor.test(TextLength, Distance, method="spearman")$p.value,
    .groups = "drop"
  ) %>%
  mutate(Query = factor(Query, levels = medical_condition_levels))

textlength_vs_distance = left_join(textlength_vs_distance, textlength_vs_distance_rho, by = "Query")

ggplot(textlength_vs_distance, aes(x = TextLength, y = Distance)) +
  geom_point() +
  facet_wrap(vars(Query)) +
  geom_text(data = textlength_vs_distance_rho, aes(label = paste0(sprintf("rho = %.2f", spearman_rho), ", ", sprintf("p = %.2f", spearman_p)), x = Inf, y = Inf), 
            hjust = 1.1, vjust = 4, check_overlap = TRUE, size=4.5) +
  xlab("Text length (# of characters)") +
  theme_bw(base_size = 16)

ggsave("Figures/Gemma_TextLength_vs_Distance.pdf", width=11, height=9, unit="in")

########################################################################
# Summarize top non-Gemma candidates according to the models.
#   These findings are based on my own evaluation, not the curators'.
########################################################################

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

# FYI: Uncomment this only if you want to overwrite the XLSX file.
#write_xlsx(top_nongemma_candidates_table, "Tables/Top_NonGemma_Candidates.xlsx")

top_nongemma_candidates_excel = readxl::read_xlsx("Tables/Top_NonGemma_Candidates.xlsx") %>%
  mutate(`Relevant to medical condition` = factor(`Relevant to medical condition`, levels=c("Yes", "Maybe", "No"))) %>%
  mutate(Query = factor(Query, levels = medical_condition_levels))

pull(top_nongemma_candidates_excel, `Relevant to medical condition`) %>%
  table() %>%
  print()

group_by(top_nongemma_candidates_excel, Query, `Relevant to medical condition`) %>%
  dplyr::summarize(Count = n()) %>%
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

# filter(top_nongemma_candidates_excel, `Relevant to medical condition` == "Yes") %>%
#   pivot_longer(cols=c(`Primary sample(s)`, `Cell line(s)`, `Xenograft(s)`), names_to = "Sample type", values_to = "Response") %>%
#   filter(Response == "Yes") %>%
#   mutate(`Sample type` = factor(`Sample type`, levels = c("Primary sample(s)", "Cell line(s)", "Xenograft(s)"))) %>%
#   group_by(Query, `Sample type`) %>%
#   dplyr::summarize(Count = n()) %>%
#   ggplot(aes(x = Query, y = Count, fill = `Sample type`)) +
#   geom_col() +
#   coord_flip() +
#   xlab("") +
#   scale_fill_brewer(palette = "Set2") +
#   theme_bw(base_size=16)
# 
# ggsave("Figures/NonGemma_SampleType.pdf", width=11, height=9, unit="in")

########################################################################
# Summarize top manual candidates and compare against top model candidates.
#   These findings are based on my own evaluation, not the curators'.
########################################################################

manual_search_items %>%
  filter(Search_Type == "Tag ontology term plus synonyms") %>%
  filter(In_Gemma == "No") %>%
  arrange(Rank) %>%
  dplyr::select(-Search_Type, -In_Gemma, -Matches_Gemma, -Rank) %>%
  group_by(Query) %>%
  slice_head(n = 50) %>%
  dplyr::rename(`Series title` = Series_Title) %>%
  dplyr::rename(`Series summary` = Series_Summary) %>%
  dplyr::rename(`Series overall design` = Series_Overall_Design) %>%
  mutate(`Comment(s)` = "") %>%
  mutate(`Relevant to medical condition` =  "No") -> top_manual_candidates_table

# Only execute this code when you want to create the Excel file.
#write_xlsx(top_manual_candidates_table, "Tables/Top_Manual_Candidates.xlsx")

top_manual_candidates_excel = readxl::read_xlsx("Tables/Top_Manual_Candidates.xlsx") %>%
  mutate(`Relevant to medical condition` = factor(`Relevant to medical condition`, levels=c("Yes", "Maybe", "No"))) %>%
  mutate(Query = factor(Query, levels = medical_condition_levels)) %>%
  mutate(Model = "Manual GEO search")

combined = dplyr::select(top_nongemma_candidates_excel, Model, Query, Series, `Relevant to medical condition`) %>%
  bind_rows(dplyr::select(top_manual_candidates_excel, Model, Query, Series, `Relevant to medical condition`))

combined_counts = group_by(combined, Model, Query) %>%
  dplyr::summarize(Total = n())

plot_data = group_by(combined, Model, Query, `Relevant to medical condition`) %>%
  dplyr::summarize(Count = n()) %>%
  inner_join(combined_counts) %>%
  mutate(Percentage = round(100 * Count / Total, 1))

ggplot(plot_data, aes(x = `Relevant to medical condition`, y = Percentage, fill = Model)) +
    geom_col(position = "dodge2") +
    ylab("Percentage") +
    facet_wrap(vars(Query)) +
    theme_bw(base_size = 14) +
    scale_fill_brewer(palette = "Set2") +
    labs(fill = "")

ggsave("Figures/NonGemma_vs_Manual_Precision.pdf", width=11, height=9, unit="in")

filter(combined, `Relevant to medical condition` == "Yes") %>%
  group_by(Query, Series) %>%
  dplyr::summarize(Num_Overlapping = n()) %>%
  group_by(Num_Overlapping) %>%
  dplyr::summarize(Count = n()) %>%
  print()

# Num_Overlapping Count
# <int> <int>
# 1               1   252
# 2               2    57 (18.4%)


########################################################################
# Randomly select subsets for UBC curators with 10% overlap.
########################################################################

retrieve_top_bottom = function(tbl, query) {
  tbl2 = filter(tbl, Query == query) %>%
    slice_sample(prop = 1)
  
  n = ceiling(nrow(tbl2) / 2 + 5)
  
  top = head(tbl2, n = n)
  bottom = tail(tbl2, n = n)
  
  return(list("top" = top, "bottom" = bottom))
}

set.seed(0)

top_nongemma_candidates_ubc1 = NULL
top_nongemma_candidates_ubc2 = NULL

top_nongemma_candidates_table_tmp = top_nongemma_candidates_table %>%
  dplyr::select(-Model, -`Score`, -`Comment(s)`, -`Relevant to medical condition`, -`Relevant to related condition(s)`, -`Primary sample(s)`, -`Cell line(s)`, -`Xenograft(s)`)

for (query in unique(pull(top_nongemma_candidates_table_tmp, Query))) {
  top_bottom = retrieve_top_bottom(top_nongemma_candidates_table_tmp, query)
  
  top_nongemma_candidates_ubc1 = bind_rows(top_nongemma_candidates_ubc1, top_bottom$top)
  top_nongemma_candidates_ubc2 = bind_rows(top_nongemma_candidates_ubc2, top_bottom$bottom)
}

# Randomly shuffle and then modify the columns to be consistent with the pilot.
finalize_subset_for_ubc = function(tbl, file_name1, file_name2) {
  tbl = slice_sample(tbl, prop = 1)
  
  for (column_name in medical_condition_levels) {
    tbl[[column_name]] = "No"
  }

  tbl[["Investigate more"]] = "No"
  tbl[["Optional comments"]] = ""
  
  # Save a copy with Query and one without
  write_xlsx(tbl, file_name1)

  dplyr::select(tbl, -Query) %>%
    write_xlsx(file_name2)
}

finalize_subset_for_ubc(top_nongemma_candidates_ubc1, "Tables/Top_Candidates_EvaluationA_Curator1_WithMedicalCondition.xlsx", "Tables/Top_Candidates_EvaluationA_Curator1.xlsx")
finalize_subset_for_ubc(top_nongemma_candidates_ubc2, "Tables/Top_Candidates_EvaluationA_Curator2_WithMedicalCondition.xlsx", "Tables/Top_Candidates_EvaluationA_Curator2.xlsx")

#########

top_manual_candidates_ubc1 = NULL
top_manual_candidates_ubc2 = NULL

top_manual_candidates_table_tmp = top_manual_candidates_table %>%
  ungroup() %>%
  dplyr::select(-`Comment(s)`, -`Relevant to medical condition`)

for (query in unique(pull(top_manual_candidates_table_tmp, Query))) {
  top_bottom = retrieve_top_bottom(top_manual_candidates_table_tmp, query)

  top_manual_candidates_ubc1 = bind_rows(top_manual_candidates_ubc1, top_bottom$top)
  top_manual_candidates_ubc2 = bind_rows(top_manual_candidates_ubc2, top_bottom$bottom)
}

finalize_subset_for_ubc(top_manual_candidates_ubc1, "Tables/Top_Candidates_EvaluationB_Curator1_WithMedicalCondition.xlsx", "Tables/Top_Candidates_EvaluationB_Curator1.xlsx")
finalize_subset_for_ubc(top_manual_candidates_ubc2, "Tables/Top_Candidates_EvaluationB_Curator2_WithMedicalCondition.xlsx", "Tables/Top_Candidates_EvaluationB_Curator2.xlsx")

########################################################################
# Summarize top manual candidates and compare against top model candidates.
#   These findings are based on my own evaluation, not the curators'.
########################################################################

