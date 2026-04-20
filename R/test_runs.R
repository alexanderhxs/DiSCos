library(pracma)
library(parallel)
library(stats)
library(maps)
library(data.table)
library(CVXR)
#library(DiSCos)
library(ggplot2)

source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCo.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCo_iter.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCo_per.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCo_per_iter.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCo_CI.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCo_weights_reg.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCo_bc.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/utils.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCo_mixture.R")
source("C:/Dokumente/Studium/1. Master Thesis/DiSCos/R/DiSCoTEA.R")


load("C:/Dokumente/Studium/1. Master Thesis/DiSCos/data/dube.rda")

head(dube)

id_col.target <- 2
t0 <- 2003

df <- copy(dube)
disco <- DiSCo(df, id_col.target, t0, M=1000, G = 100, num.cores = 5, permutation = TRUE, CI = TRUE, boots = 100, graph = TRUE, simplex=TRUE, seed=NULL, q_max=0.9, mixture=FALSE)
discot = DiSCoTEA(disco, agg = 'cdfDiff')

lot_fit_quantiles_gg(disco, show_controls = FALSE)

get_discrete_data <- function(sample_size, num_controls, dist_control = 3, dist_target = 4) {

  # Hilfsfunktion für den Binomial-Mix
  draw_mix_binom <- function(n, n_comp, n_trials, p_probs) {
    # R ist 1-basiert, daher ziehen wir von 1 bis n_comp
    c <- sample(1:n_comp, size = n, replace = TRUE)
    # rbinom ist in R bereits vektorisiert und nimmt Vektoren für size und prob an
    return(rbinom(n = n, size = n_trials[c], prob = p_probs[c]))
  }

  n_trials_t <- sample(1:19, size = dist_target, replace = TRUE)
  p_probs_t  <- runif(n = dist_target, min = 0.1, max = 0.9)

  target_pre  <- draw_mix_binom(sample_size, dist_target, n_trials_t, p_probs_t)
  target_post <- draw_mix_binom(sample_size, dist_target, n_trials_t, p_probs_t)

  # In R bauen wir Vektoren direkt zu einem Dataframe zusammen (viel schneller als list comprehensions)
  df_target <- data.frame(
    id_col = 0,
    time_col = rep(c(9998, 9999), each = sample_size),
    y_col = c(target_pre, target_post),
    stringsAsFactors = FALSE
  )

  # Liste initiieren, um alle Dataframes am Ende zusammenzufügen
  df_list <- list()
  df_list[[1]] <- df_target

  # Controls (jede Unit hat ihren eigenen Binomial-Mix)
  if (num_controls > 0) {
    for (i in 1:num_controls) {
      n_trials_c <- sample(1:19, size = dist_control, replace = TRUE)
      p_probs_c  <- runif(n = dist_control, min = 0.1, max = 0.9)

      c_pre  <- draw_mix_binom(sample_size, dist_control, n_trials_c, p_probs_c)
      c_post <- draw_mix_binom(sample_size, dist_control, n_trials_c, p_probs_c)

      df_c <- data.frame(
        id_col = as.numeric(i),
        time_col = rep(c(9998, 9999), each = sample_size),
        y_col = c(c_pre, c_post),
        stringsAsFactors = FALSE
      )

      df_list[[i + 1]] <- df_c
    }
  }

  result_df <- do.call(rbind, df_list)

  return(result_df)
}

get_continuous_data <- function(sample_size, num_controls, dist_control = 3, dist_target = 4) {

  # Hilfsfunktion für den Gaussian-Mix
  draw_mix <- function(n, n_comp, m, s) {
    # Wir ziehen zufällig die Indizes 1 bis n_comp
    c <- sample(1:n_comp, size = n, replace = TRUE)
    # rnorm ist in R vektorisiert und zieht für jeden Eintrag den passenden mean und sd
    return(rnorm(n = n, mean = m[c], sd = s[c]))
  }

  # Target (Mix aus dist_target Gauss-Kurven, standardmäßig 4)
  means_t <- runif(dist_target, min = -10, max = 10)
  variances_t <- runif(dist_target, min = 0.5, max = 6)
  stdevs_t <- sqrt(variances_t)

  target_pre <- draw_mix(sample_size, dist_target, means_t, stdevs_t)
  target_post <- draw_mix(sample_size, dist_target, means_t, stdevs_t)

  # Dataframe für das Target erstellen
  df_target <- data.frame(
    id_col = 0,
    time_col = rep(c(9998, 9999), each = sample_size),
    y_col = c(target_pre, target_post),
    stringsAsFactors = FALSE
  )

  # Liste initiieren, um alle Dataframes am Ende schnell zusammenzufügen
  df_list <- list()
  df_list[[1]] <- df_target

  # Controls (jede Unit hat ihren eigenen Gauss-Mix)
  if (num_controls > 0) {
    for (i in 1:num_controls) {
      means_c <- runif(dist_control, min = -10, max = 10)
      variances_c <- runif(dist_control, min = 0.5, max = 6)
      stdevs_c <- sqrt(variances_c)

      c_pre <- draw_mix(sample_size, dist_control, means_c, stdevs_c)
      c_post <- draw_mix(sample_size, dist_control, means_c, stdevs_c)

      df_c <- data.frame(
        id_col = i,
        time_col = rep(c(9998, 9999), each = sample_size),
        y_col = c(c_pre, c_post),
        stringsAsFactors = FALSE
      )

      df_list[[i + 1]] <- df_c
    }
  }

  # Alle einzelnen Dataframes zu einem großen Dataframe zusammenfügen
  result_df <- do.call(rbind, df_list)

  return(result_df)
}



lot_fit_quantiles_gg <- function(fit_synth, show_controls = FALSE) {

  period_res <- fit_synth$results.periods[[1]]

  x_grid <- fit_synth$evgrid
  target_quantiles <- period_res$target$quantiles
  disco_quantiles <- period_res$DiSCo$quantile
  # Basis-Dataframe für Target und DSC
  df <- data.frame(
    x = x_grid,
    Target = target_quantiles,
    DSC = disco_quantiles
  )

  # Plot initialisieren
  p <- ggplot()

  # Controls plotten (falls aktiviert)
  if (show_controls) {
    controls_mat <- period_res$controls$quantiles
    n_controls <- ncol(controls_mat)

    # In ggplot baut man das eleganter mit einer kleinen Schleife oder über Pivot
    for (i in 1:n_controls) {
      control_df <- data.frame(x = x_grid, y = controls_mat[, i])
      p <- p + geom_line(data = control_df, aes(x = x, y = y, color = "Controls"),
                         linewidth = 0.5, linetype = "dashed")
    }
  }

  # Target und DSC oben drauf legen
  p <- p +
    geom_line(data = df, aes(x = x, y = Target, color = "Target"), linewidth = 1.5) +
    geom_line(data = df, aes(x = x, y = DSC, color = "DSC"), linewidth = 1.5) +

    # Farben und Legende steuern
    scale_color_manual(
      name = NULL,
      values = c("Target" = "black", "DSC" = "red", "Controls" = "grey")
    ) +

    # Achsenlimits
    coord_cartesian(xlim = c(-0.02, 1.02)) +

    # expression() rendert das mathematische LaTeX-Format in R
    labs(x = "x", y = expression(F^{-1}(x))) +

    # theme_bw() gibt die Achsenumrandung und das Raster
    theme_bw() +
    theme(
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      legend.text = element_text(size = 12),
      # Legende unten rechts IN den Plot schieben
      legend.position = c(0.98, 0.02),
      legend.justification = c(1, 0),
      legend.background = element_rect(color = "black", linewidth = 0.5)
    )

  print(p)
}

sample_size = 1000
num_controls = 30
set.seed(123)
# --- Testaufruf ---
df <- get_discrete_data(sample_size = sample_size, num_controls = num_controls)
id_col.target <- 0
t0 <- 9999
fit_synth = DiSCo(df, id_col.target=id_col.target,
                  t0=t0,
                  G = 1000,
                  M= 1000,
                  num.cores = 1,
                  permutation = FALSE,
                  CI = FALSE,
                  boots = 1000,
                  graph = FALSE,
                  q_min = 0,
                  q_max=1,
                  seed=1,
                  simplex=TRUE,
                  mixture=TRUE)

lot_fit_quantiles_gg(disco, show_controls = FALSE)

df <- get_continuous_data(sample_size = sample_size, num_controls = num_controls)
id_col.target <- 0
t0 <- 9999
fit_synth = DiSCo(df, id_col.target=id_col.target,
                  t0=t0,
                  G = 1000,
                  M= 1000,
                  num.cores = 1,
                  permutation = FALSE,
                  CI = FALSE,
                  boots = 1000,
                  graph = FALSE,
                  q_min = 0,
                  q_max=1,
                  seed=1,
                  simplex=TRUE,
                  mixture=TRUE)

lot_fit_quantiles_gg(fit_synth, show_controls = FALSE)




