# src/r_analysis.R
# Tema Avanzado - Scripts en R con tidyverse y ggplot2

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

cat("=== Analisis estadistico en R - Air Quality ===\n\n")

candidates <- c(
  "data/airquality_processed.csv",
  "../data/airquality_processed.csv",
  "proyecto_air/data/airquality_processed.csv"
)
data_path <- NULL
for (p in candidates) {
  if (file.exists(p)) { data_path <- p; break }
}
if (is.null(data_path)) stop("No se encontro airquality_processed.csv")

df <- read_csv(data_path, show_col_types = FALSE)
cat("Dataset cargado:", nrow(df), "filas\n\n")

# 1. CO por hora punta
cat("-- CO por hora punta --\n")
df %>%
  mutate(punta = ifelse(is_rush_hour == 1, "Hora punta", "Fuera de punta")) %>%
  group_by(punta) %>%
  summarise(media = mean(`CO(GT)`, na.rm=TRUE),
            mediana = median(`CO(GT)`, na.rm=TRUE),
            n = n(), .groups="drop") %>%
  print()

# 2. CO por temperatura
cat("\n-- CO por temperatura --\n")
df %>%
  filter(!is.na(temp_cat)) %>%
  group_by(temp_cat) %>%
  summarise(media_CO  = mean(`CO(GT)`, na.rm=TRUE),
            media_NOx = mean(`NOx(GT)`, na.rm=TRUE),
            n = n(), .groups="drop") %>%
  arrange(desc(media_CO)) %>%
  print()

# 3. Correlaciones
cat("\n-- Correlacion con CO(GT) --\n")
num_vars <- c("PT08.S1(CO)","C6H6(GT)","NOx(GT)","NO2(GT)","T","RH","AH","sensor_mean")
num_vars <- num_vars[num_vars %in% names(df)]
df %>%
  select(all_of(c(num_vars, "CO(GT)"))) %>%
  drop_na() %>%
  summarise(across(all_of(num_vars), ~cor(.x, `CO(GT)`, use="complete.obs"))) %>%
  pivot_longer(everything(), names_to="variable", values_to="correlacion") %>%
  arrange(desc(abs(correlacion))) %>%
  print()

# 4. Modelo lineal
cat("\n-- Regresion lineal lm() --\n")
df_lm <- df %>%
  select(`CO(GT)`, `PT08.S1(CO)`, `C6H6(GT)`, `NOx(GT)`, T, RH, AH, hour, month) %>%
  drop_na()
lm_model <- lm(`CO(GT)` ~ ., data = df_lm)
cat(sprintf("R2 ajustado: %.4f\n", summary(lm_model)$adj.r.squared))

# 5. Graficos ggplot2
plots_dir <- file.path(dirname(data_path), "plots")
dir.create(plots_dir, showWarnings=FALSE, recursive=TRUE)

p1 <- df %>%
  filter(!is.na(temp_cat)) %>%
  ggplot(aes(x=temp_cat, y=`CO(GT)`, fill=temp_cat)) +
  geom_boxplot(outlier.alpha=0.2) +
  scale_fill_brewer(palette="RdYlBu", direction=-1) +
  labs(title="CO por categoria de temperatura", subtitle="Analisis con R + ggplot2",
       x="Temperatura", y="CO (mg/m3)") +
  theme_minimal() +
  theme(legend.position="none", plot.title=element_text(face="bold"))
ggsave(file.path(plots_dir, "12_r_temp_boxplot.png"), p1, width=7, height=5, dpi=130)
cat("Guardado: 12_r_temp_boxplot.png\n")

p2 <- df %>%
  group_by(hour) %>%
  summarise(media=mean(`CO(GT)`, na.rm=TRUE), .groups="drop") %>%
  ggplot(aes(x=hour, y=media)) +
  geom_col(fill="#E53935", alpha=0.8) +
  geom_smooth(se=FALSE, color="black", linewidth=0.8) +
  labs(title="CO medio por hora del dia", subtitle="ggplot2",
       x="Hora", y="CO medio (mg/m3)") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"))
ggsave(file.path(plots_dir, "13_r_co_hora.png"), p2, width=9, height=5, dpi=130)
cat("Guardado: 13_r_co_hora.png\n")

cat("\n[R] Analisis completado.\n")
