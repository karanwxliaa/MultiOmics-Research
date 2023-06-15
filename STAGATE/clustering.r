# Set the CRAN mirror URL
cran_mirror <- "https://cran.rstudio.com/"

# Set the desired package name
package_name <- "mclust"

# Check if the package is already installed
if (!requireNamespace(package_name, quietly = TRUE)) {
  # Install the package from the specified CRAN mirror
  install.packages(package_name, repos = cran_mirror)
}

# Load the package
library(mclust)

# Rest of your code here...
mclust_R <- function(adata, num_cluster=7, modelNames = "EEE", used_obsm = "STAGATE", random_seed = 2020){
  set.seed(random_seed)
  library(mclust)
  # Convert numpy array to R matrix
  adata_matrix <- t(as.matrix(adata$obsm[[used_obsm]]))
  # Set the random seed
  set.seed(random_seed)
  # Run the Mclust function
  mclust_res <- Mclust(adata_matrix, G = num_cluster, modelNames = modelNames)
  # Get the cluster assignments
  cluster_labels <- mclust_res$classification
  # Assign the cluster labels to adata$obs[['mclust']]
  adata$obs[['mclust']] <- factor(cluster_labels, levels = 1:num_cluster)
  return(adata)
}

