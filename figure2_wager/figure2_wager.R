library(tree)
library(ggplot2)
library(parallel)
library(pbapply)

# Constants
NOISE_VARIANCE <- 0.25
MAX_LEAF_NODES <- 5

step_function <- function(x) {
  ifelse(x < 0.35, 0,
         ifelse(x < 0.45, 0.7,
                ifelse(x < 0.55, 1.4,
                       ifelse(x < 0.65, 0.7, 0))))
}

generate_data <- function(x_points, y_true, 
                          noise_variance = NOISE_VARIANCE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  noise <- rnorm(length(x_points), 0, sqrt(noise_variance))
  y_true + noise
}

create_bootstrap_indices_and_Nbi <- function(n_data_points, n_bootstrap, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  indices_list <- replicate(n_bootstrap, sample(n_data_points, n_data_points, replace = TRUE))
  counts <- matrix(0, n_bootstrap, n_data_points)
  for (x_i in 1:n_data_points) {
    counts[, x_i] <- colSums(indices_list == x_i)
  }
  list(indices_list = indices_list, counts = counts)
}

bagging_decision_trees <- function(x_points, y_noisy, n_bootstrap, max_leaf_nodes = MAX_LEAF_NODES, seed = NULL) {
  n_data_points <- length(x_points)
  tree_predictions_b <- matrix(0, n_bootstrap, n_data_points)
  bootstrap_data <- create_bootstrap_indices_and_Nbi(n_data_points, n_bootstrap, seed)
  indices_list <- bootstrap_data$indices_list
  N_bi <- bootstrap_data$counts
  
  for (b in 1:n_bootstrap) {
    data_bootstrap <- data.frame(x = x_points[indices_list[, b]], y = y_noisy[indices_list[, b]])
    model <- tree(y ~ x, data = data_bootstrap)
    prune_model <- prune.tree(model, best = max_leaf_nodes)
    tree_predictions_b[b, ] <- predict(prune_model, newdata = data.frame(x = x_points))
  }
  
  list(tree_predictions_b = tree_predictions_b, N_bi = N_bi)
}

inf_JK_bagged_variance <- function(N_bi, tree_predictions_b, chunk_size = 250) {
  n_bootstrap <- nrow(N_bi)
  n_data_points <- ncol(N_bi)
  n_preds <- ncol(tree_predictions_b)
  
  N_star_mean <- matrix(1, n_bootstrap, n_preds)
  T_N_star_mean <- colMeans(tree_predictions_b)
  
  cov_matrix <- matrix(0, n_data_points, n_preds)
  
  for (start in seq(1, n_bootstrap, by = chunk_size)) {
    end <- min(start + chunk_size - 1, n_bootstrap)
    chunk_N_bi <- N_bi[start:end, ]
    chunk_tree_predictions_b <- tree_predictions_b[start:end, ]
    
    # Calculate the covariance matrix for the chunk
    chunk_cov_matrix <- colSums((t(chunk_N_bi - N_star_mean[start:end, ]) %*% sweep(chunk_tree_predictions_b, 2, T_N_star_mean, FUN = "-"))^2)
    cov_matrix <- cov_matrix + chunk_cov_matrix
  }
  
  cov_matrix <- cov_matrix / (n_bootstrap - 1)^2
  
  bias_corr <- apply(tree_predictions_b, 2, var) * ((n_data_points - 1) * n_bootstrap) / (n_bootstrap - 1)^2
  bagged_inf_jackknife_est <- cov_matrix - bias_corr
  
  bagged_inf_jackknife_est[,1]
}

simulate_bagging_and_variance <- function(x_points, y_true, n_bootstrap, simulation_index, seed) {
  y_noisy <- generate_data(x_points, y_true, NOISE_VARIANCE, seed + simulation_index)
  bagging_result <- bagging_decision_trees(x_points, y_noisy, n_bootstrap, seed = seed + simulation_index)
  tree_predictions_b <- bagging_result$tree_predictions_b
  N_bi <- bagging_result$N_bi
  bagged_predictions <- colMeans(tree_predictions_b)
  est_variances <- inf_JK_bagged_variance(N_bi, tree_predictions_b)
  
  list(bagged_predictions = bagged_predictions, est_variances = est_variances)
}

save_results_png <- function(x_points, true_variances, est_variances_mean, est_variances_std, n_data_points, n_simulations, n_bootstrap, seed) {
  df <- data.frame(
    x = x_points,
    true_variances = true_variances,
    est_variances_mean = est_variances_mean,
    lower_bound = est_variances_mean - est_variances_std,
    upper_bound = est_variances_mean + est_variances_std
  )
  
  plot <- ggplot(df, aes(x = x)) +
    geom_line(aes(y = true_variances, color = "True Variance")) +
    geom_line(aes(y = est_variances_mean, color = "Mean Est. Variance")) +
    geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound), alpha = 0.2, fill = "blue") +
    labs(title = "True Variance of Bagged Predictions Across Simulated Datasets", x = "x", y = "Variance") +
    annotate("text", x = 0.05, y = 0.05, label = paste("data_points =", n_data_points, "\nsimulations =", n_simulations, "\nbootstrap(B) =", n_bootstrap),
             hjust = 0, vjust = 1, size = 5) +
    theme(legend.position = "bottom") +
    ylim(-0.02, 0.06)  # Set y-axis limits
  
  ggsave(paste0("figure2_wager/figures/Rfigure2_wager_seed_", seed, "_nx", n_data_points, "_nB", n_bootstrap, "_", as.integer(Sys.time()), ".png"), plot = plot)
}


main <- function() {
  n_data_points <- 500
  n_simulations <- 1000
  n_bootstrap <- 500
  seed <- 62
  
  x_points <- seq(0, 1, length.out = n_data_points)
  y_true <- step_function(x_points)
  
  cl <- makeCluster(detectCores() - 1) # Create a cluster using all but one core
  clusterEvalQ(cl, {
    library(tree)
    library(ggplot2)
  })
  clusterExport(cl, c("generate_data", "bagging_decision_trees", "create_bootstrap_indices_and_Nbi",
                      "inf_JK_bagged_variance", "simulate_bagging_and_variance", "NOISE_VARIANCE",
                      "MAX_LEAF_NODES", "step_function", "x_points", "y_true", "n_bootstrap", "seed")) # Export functions and constants to the cluster
  
  results <- pblapply(1:n_simulations, function(i) {
    simulate_bagging_and_variance(x_points, y_true, n_bootstrap, i, seed)
  }, cl = cl)
  
  stopCluster(cl) # Stop the cluster
  
  bagged_predictions_all <- matrix(unlist(lapply(results, `[[`, "bagged_predictions")), nrow = n_simulations, byrow = TRUE)
  est_variances_all <- matrix(unlist(lapply(results, `[[`, "est_variances")), nrow = n_simulations, byrow = TRUE)
  
  true_variances <- apply(bagged_predictions_all, 2, var)
  est_variances_mean <- colMeans(est_variances_all)
  est_variances_std <- apply(est_variances_all, 2, sd)
  
  cat("Mean true variance:", round(mean(true_variances), 10), "\n")
  cat("Mean estimated variance:", round(mean(est_variances_mean), 10), "\n")
  cat("Min estimated variance:", round(min(est_variances_all), 10), "\n")
  
  save_results_png(x_points, true_variances, est_variances_mean, est_variances_std, n_data_points, n_simulations, n_bootstrap, seed)
}

# Ensure variables are defined in the global environment
n_data_points <- 300
n_simulations <- 10
n_bootstrap <- 300
seed <- 62

x_points <- seq(0, 1, length.out = n_data_points)
y_true <- step_function(x_points)

start_time <- Sys.time()
main()
cat("--- runtime:", round(difftime(Sys.time(), start_time, units = "mins"), 2), "minutes ---\n")