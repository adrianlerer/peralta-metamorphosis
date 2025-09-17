# Argentine Legal Evolution Dataset - Descriptive Statistics
# Version 1.0.0-beta
# Author: Adrian Lerer
# Date: September 2025

# Load required libraries
library(tidyverse)
library(lubridate)
library(ggplot2)
library(knitr)
library(corrplot)
library(survival)

# Set working directory and load data
setwd(".")  # Adjust path as needed

# Load datasets
evolution_cases <- read_csv("evolution_cases.csv", 
                          col_types = cols(
                            fecha_inicio = col_date(format = "%Y-%m-%d"),
                            fecha_fin = col_date(format = "%Y-%m-%d"),
                            velocidad_cambio_dias = col_integer(),
                            supervivencia_anos = col_integer(),
                            mutaciones_identificadas = col_integer()
                          ))

velocity_metrics <- read_csv("velocity_metrics.csv",
                           col_types = cols(
                             value = col_double(),
                             confidence_level = col_factor(levels = c("Low", "Medium", "High"))
                           ))

transplants_tracking <- read_csv("transplants_tracking.csv",
                               col_types = cols(
                                 introduction_date = col_date(format = "%Y-%m-%d"),
                                 success_level = col_factor(levels = c("Failed", "Partial", "Developing", "High")),
                                 survival_years = col_integer(),
                                 mutations_count = col_integer()
                               ))

crisis_periods <- read_csv("crisis_periods.csv",
                         col_types = cols(
                           start_date = col_date(format = "%Y-%m-%d"),
                           end_date = col_date(format = "%Y-%m-%d"),
                           severity_level = col_factor(levels = c("Low", "Medium", "High", "Very High", "Extreme")),
                           legal_changes_count = col_integer(),
                           acceleration_factor = col_double()
                         ))

innovations_exported <- read_csv("innovations_exported.csv",
                               col_types = cols(
                                 origin_date = col_date(format = "%Y-%m-%d"),
                                 success_level = col_factor(levels = c("Low", "Medium", "High", "Very High")),
                                 regional_influence = col_factor(levels = c("Low", "Medium", "High", "Very High"))
                               ))

# ================================================================================
# 1. EVOLUTION CASES ANALYSIS
# ================================================================================

cat("=== ARGENTINE LEGAL EVOLUTION DATASET - DESCRIPTIVE STATISTICS ===\n\n")

# Basic dataset info
cat("1. DATASET OVERVIEW\n")
cat("==================\n")
cat(sprintf("Evolution Cases: %d observations\n", nrow(evolution_cases)))
cat(sprintf("Velocity Metrics: %d observations\n", nrow(velocity_metrics)))
cat(sprintf("Transplants Tracked: %d observations\n", nrow(transplants_tracking)))
cat(sprintf("Crisis Periods: %d observations\n", nrow(crisis_periods)))
cat(sprintf("Innovations Exported: %d observations\n", nrow(innovations_exported)))
cat(sprintf("Time Coverage: %s to %s\n", 
    min(evolution_cases$fecha_inicio, na.rm = TRUE),
    max(evolution_cases$fecha_fin, na.rm = TRUE)))
cat("\n")

# Evolution cases summary
cat("2. LEGAL EVOLUTION PATTERNS\n")
cat("============================\n")

# By legal area
area_summary <- evolution_cases %>%
  count(area_derecho, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Cases by Legal Area:\n")
print(kable(area_summary, col.names = c("Legal Area", "Count", "Percentage")))
cat("\n")

# By selection type
selection_summary <- evolution_cases %>%
  count(tipo_seleccion, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Cases by Selection Type:\n")
print(kable(selection_summary, col.names = c("Selection Type", "Count", "Percentage")))
cat("\n")

# By origin
origin_summary <- evolution_cases %>%
  count(origen, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Cases by Origin:\n")
print(kable(origin_summary, col.names = c("Origin", "Count", "Percentage")))
cat("\n")

# By success level
success_summary <- evolution_cases %>%
  count(exito, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Cases by Success Level:\n")
print(kable(success_summary, col.names = c("Success Level", "Count", "Percentage")))
cat("\n")

# ================================================================================
# 2. TEMPORAL PATTERNS ANALYSIS
# ================================================================================

cat("3. TEMPORAL EVOLUTION PATTERNS\n")
cat("===============================\n")

# Add decade variable
evolution_cases <- evolution_cases %>%
  mutate(
    decade_inicio = floor(year(fecha_inicio) / 10) * 10,
    duracion_anos = as.numeric(fecha_fin - fecha_inicio) / 365.25
  )

# Cases by decade
decade_summary <- evolution_cases %>%
  count(decade_inicio, sort = TRUE) %>%
  filter(!is.na(decade_inicio))

cat("Legal Innovations by Decade of Introduction:\n")
print(kable(decade_summary, col.names = c("Decade", "Count")))
cat("\n")

# Survival statistics
survival_stats <- evolution_cases %>%
  filter(!is.na(supervivencia_anos)) %>%
  summarise(
    mean_survival = round(mean(supervivencia_anos), 1),
    median_survival = round(median(supervivencia_anos), 1),
    min_survival = min(supervivencia_anos),
    max_survival = max(supervivencia_anos),
    sd_survival = round(sd(supervivencia_anos), 1)
  )

cat("Survival Statistics (Years):\n")
print(kable(t(survival_stats), col.names = c("Statistic", "Value")))
cat("\n")

# Velocity statistics
velocity_stats <- evolution_cases %>%
  filter(!is.na(velocidad_cambio_dias), velocidad_cambio_dias > 0) %>%
  summarise(
    mean_days = round(mean(velocidad_cambio_dias), 0),
    median_days = round(median(velocidad_cambio_dias), 0),
    mean_years = round(mean(velocidad_cambio_dias)/365.25, 1),
    median_years = round(median(velocidad_cambio_dias)/365.25, 1)
  )

cat("Evolution Velocity Statistics:\n")
print(kable(t(velocity_stats), col.names = c("Metric", "Value")))
cat("\n")

# ================================================================================
# 3. MUTATION PATTERNS ANALYSIS
# ================================================================================

cat("4. MUTATION AND ADAPTATION PATTERNS\n")
cat("====================================\n")

# Mutation rate statistics
mutation_stats <- evolution_cases %>%
  filter(!is.na(mutaciones_identificadas)) %>%
  summarise(
    cases_with_mutations = sum(mutaciones_identificadas > 0),
    total_cases = n(),
    mutation_rate = round(cases_with_mutations/total_cases*100, 1),
    mean_mutations = round(mean(mutaciones_identificadas), 2),
    max_mutations = max(mutaciones_identificadas),
    median_mutations = median(mutaciones_identificadas)
  )

cat("Mutation Statistics:\n")
print(kable(t(mutation_stats), col.names = c("Metric", "Value")))
cat("\n")

# Mutation rate by success level
mutation_by_success <- evolution_cases %>%
  filter(!is.na(mutaciones_identificadas), !is.na(exito)) %>%
  group_by(exito) %>%
  summarise(
    cases = n(),
    mean_mutations = round(mean(mutaciones_identificadas), 2),
    mutation_rate = round(sum(mutaciones_identificadas > 0)/n()*100, 1),
    .groups = 'drop'
  )

cat("Mutations by Success Level:\n")
print(kable(mutation_by_success))
cat("\n")

# ================================================================================
# 4. CRISIS IMPACT ANALYSIS
# ================================================================================

cat("5. CRISIS ACCELERATION PATTERNS\n")
cat("================================\n")

# Crisis severity distribution
crisis_severity <- crisis_periods %>%
  count(severity_level, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Crisis by Severity Level:\n")
print(kable(crisis_severity, col.names = c("Severity", "Count", "Percentage")))
cat("\n")

# Acceleration factor statistics
acceleration_stats <- crisis_periods %>%
  filter(!is.na(acceleration_factor)) %>%
  summarise(
    mean_acceleration = round(mean(acceleration_factor), 2),
    median_acceleration = round(median(acceleration_factor), 2),
    min_acceleration = round(min(acceleration_factor), 2),
    max_acceleration = round(max(acceleration_factor), 2),
    sd_acceleration = round(sd(acceleration_factor), 2)
  )

cat("Legal Change Acceleration During Crises:\n")
print(kable(t(acceleration_stats), col.names = c("Statistic", "Value")))
cat("\n")

# Crisis impact by severity
crisis_impact <- crisis_periods %>%
  filter(!is.na(acceleration_factor), !is.na(severity_level)) %>%
  group_by(severity_level) %>%
  summarise(
    crises_count = n(),
    mean_acceleration = round(mean(acceleration_factor), 2),
    mean_legal_changes = round(mean(legal_changes_count, na.rm = TRUE), 0),
    .groups = 'drop'
  )

cat("Crisis Impact by Severity:\n")
print(kable(crisis_impact))
cat("\n")

# ================================================================================
# 5. TRANSPLANT SUCCESS ANALYSIS
# ================================================================================

cat("6. LEGAL TRANSPLANT PATTERNS\n")
cat("=============================\n")

# Transplant success rates
transplant_success <- transplants_tracking %>%
  count(success_level, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Transplant Success Distribution:\n")
print(kable(transplant_success, col.names = c("Success Level", "Count", "Percentage")))
cat("\n")

# Success by origin legal family
success_by_family <- transplants_tracking %>%
  filter(!is.na(success_level)) %>%
  count(origin_legal_family, success_level) %>%
  group_by(origin_legal_family) %>%
  mutate(
    total = sum(n),
    percentage = round(n/total*100, 1)
  ) %>%
  filter(success_level == "High") %>%
  select(origin_legal_family, high_success_count = n, total, success_rate = percentage) %>%
  arrange(desc(success_rate))

cat("High Success Rate by Legal Family Origin:\n")
print(kable(success_by_family))
cat("\n")

# Survival by success level
transplant_survival <- transplants_tracking %>%
  filter(!is.na(survival_years), !is.na(success_level)) %>%
  group_by(success_level) %>%
  summarise(
    cases = n(),
    mean_survival = round(mean(survival_years), 1),
    median_survival = round(median(survival_years), 1),
    .groups = 'drop'
  )

cat("Transplant Survival by Success Level:\n")
print(kable(transplant_survival))
cat("\n")

# ================================================================================
# 6. EXPORT SUCCESS ANALYSIS
# ================================================================================

cat("7. ARGENTINE LEGAL INNOVATION EXPORTS\n")
cat("======================================\n")

# Export success distribution
export_success <- innovations_exported %>%
  count(success_level, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Export Success Distribution:\n")
print(kable(export_success, col.names = c("Success Level", "Count", "Percentage")))
cat("\n")

# Regional influence distribution
regional_influence <- innovations_exported %>%
  count(regional_influence, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Regional Influence Distribution:\n")
print(kable(regional_influence, col.names = c("Influence Level", "Count", "Percentage")))
cat("\n")

# Most successful exports (high success + high regional influence)
successful_exports <- innovations_exported %>%
  filter(success_level %in% c("High", "Very High"), 
         regional_influence %in% c("High", "Very High")) %>%
  select(innovation_name, legal_area, success_level, regional_influence) %>%
  arrange(desc(success_level), desc(regional_influence))

cat("Most Successful Argentine Legal Exports:\n")
print(kable(successful_exports))
cat("\n")

# Export by legal area
export_by_area <- innovations_exported %>%
  count(legal_area, sort = TRUE) %>%
  mutate(percentage = round(n/sum(n)*100, 1))

cat("Exports by Legal Area:\n")
print(kable(export_by_area, col.names = c("Legal Area", "Count", "Percentage")))
cat("\n")

# ================================================================================
# 7. COMPARATIVE STATISTICS
# ================================================================================

cat("8. COMPARATIVE PATTERNS\n")
cat("========================\n")

# Evolution vs Transplants success comparison
evolution_success_rate <- evolution_cases %>%
  summarise(
    total_cases = n(),
    successful_cases = sum(exito == "Exitoso", na.rm = TRUE),
    success_rate = round(successful_cases/total_cases*100, 1)
  )

transplant_success_rate <- transplants_tracking %>%
  summarise(
    total_cases = n(),
    successful_cases = sum(success_level == "High", na.rm = TRUE),
    success_rate = round(successful_cases/total_cases*100, 1)
  )

cat("Success Rate Comparison:\n")
cat(sprintf("Endogenous Evolution Success Rate: %.1f%%\n", evolution_success_rate$success_rate))
cat(sprintf("Legal Transplant Success Rate: %.1f%%\n", transplant_success_rate$success_rate))
cat("\n")

# Survival comparison
evolution_mean_survival <- evolution_cases %>%
  filter(!is.na(supervivencia_anos)) %>%
  summarise(mean_survival = mean(supervivencia_anos)) %>%
  pull(mean_survival)

transplant_mean_survival <- transplants_tracking %>%
  filter(!is.na(survival_years)) %>%
  summarise(mean_survival = mean(survival_years)) %>%
  pull(mean_survival)

cat("Survival Comparison:\n")
cat(sprintf("Endogenous Evolution Mean Survival: %.1f years\n", evolution_mean_survival))
cat(sprintf("Legal Transplant Mean Survival: %.1f years\n", transplant_mean_survival))
cat("\n")

# ================================================================================
# 8. SUMMARY STATISTICS
# ================================================================================

cat("9. DATASET SUMMARY STATISTICS\n")
cat("==============================\n")

# Key findings summary
cat("KEY EMPIRICAL FINDINGS:\n\n")

cat(sprintf("• Total legal evolution cases documented: %d\n", nrow(evolution_cases)))
cat(sprintf("• Time span covered: %d years (%s to %s)\n", 
    year(max(evolution_cases$fecha_fin, na.rm = TRUE)) - year(min(evolution_cases$fecha_inicio, na.rm = TRUE)),
    year(min(evolution_cases$fecha_inicio, na.rm = TRUE)),
    year(max(evolution_cases$fecha_fin, na.rm = TRUE))))

cat(sprintf("• Overall success rate (endogenous): %.1f%%\n", evolution_success_rate$success_rate))
cat(sprintf("• Legal transplant success rate: %.1f%%\n", transplant_success_rate$success_rate))

cat(sprintf("• Average legal innovation survival: %.1f years\n", evolution_mean_survival))
cat(sprintf("• Crisis acceleration factor range: %.1fx to %.1fx normal velocity\n", 
    min(crisis_periods$acceleration_factor, na.rm = TRUE),
    max(crisis_periods$acceleration_factor, na.rm = TRUE)))

cat(sprintf("• Argentine innovations exported to other countries: %d\n", nrow(innovations_exported)))
cat(sprintf("• High regional influence innovations: %d\n", 
    sum(innovations_exported$regional_influence %in% c("High", "Very High"), na.rm = TRUE)))

mutation_rate_overall <- round(sum(evolution_cases$mutaciones_identificadas > 0, na.rm = TRUE) / 
                              sum(!is.na(evolution_cases$mutaciones_identificadas)) * 100, 1)
cat(sprintf("• Overall mutation rate: %.1f%% of cases show significant adaptations\n", mutation_rate_overall))

cat("\n")

cat("METHODOLOGY VALIDATION:\n")
cat("• High confidence data points: ", sum(velocity_metrics$confidence_level == "High", na.rm = TRUE), "\n")
cat("• Medium confidence data points: ", sum(velocity_metrics$confidence_level == "Medium", na.rm = TRUE), "\n")
cat("• Low confidence data points: ", sum(velocity_metrics$confidence_level == "Low", na.rm = TRUE), "\n")

# Data completeness
completeness <- evolution_cases %>%
  summarise(
    fecha_inicio_complete = round(sum(!is.na(fecha_inicio))/n()*100, 1),
    fecha_fin_complete = round(sum(!is.na(fecha_fin))/n()*100, 1),
    supervivencia_complete = round(sum(!is.na(supervivencia_anos))/n()*100, 1),
    mutaciones_complete = round(sum(!is.na(mutaciones_identificadas))/n()*100, 1)
  )

cat("\nDATA COMPLETENESS:\n")
print(kable(t(completeness), col.names = c("Field", "Completeness %")))

cat("\n=== END DESCRIPTIVE STATISTICS ===\n")

# Save summary statistics for further analysis
summary_stats <- list(
  evolution_cases_count = nrow(evolution_cases),
  transplants_count = nrow(transplants_tracking),
  crisis_periods_count = nrow(crisis_periods),
  innovations_exported_count = nrow(innovations_exported),
  evolution_success_rate = evolution_success_rate$success_rate,
  transplant_success_rate = transplant_success_rate$success_rate,
  mean_survival_evolution = evolution_mean_survival,
  mean_survival_transplant = transplant_mean_survival,
  mutation_rate = mutation_rate_overall,
  time_span_years = year(max(evolution_cases$fecha_fin, na.rm = TRUE)) - year(min(evolution_cases$fecha_inicio, na.rm = TRUE))
)

# Export summary for use in papers/presentations
write_rds(summary_stats, "summary_statistics.rds")

cat("\nSummary statistics saved to 'summary_statistics.rds'\n")
cat("Analysis complete. Use visualizations.py for graphical analysis.\n")