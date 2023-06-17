# Read the data file
data <- read.csv("C:\\Users\\KARAN\\Desktop\\MultiOmics-Research\\arch2\\protein\\data.csv", header = FALSE)
# Load the mclust package
library(mclust)

# Rest of your code here...
mclust_R <- function(data, num_cluster=8, modelNames = "EEE", used_obsm = "MY_ARCH", random_seed = 2020){ # nolint
  set.seed(random_seed)
  library(mclust)
  
  # Set the random seed
  set.seed(random_seed)
  # Run the Mclust function
  mclust_res <- Mclust(data, G = num_cluster, modelNames = modelNames)
  # Get the cluster assignments
  print(mclust_res)
  cluster_labels <- mclust_res$classification
  return(cluster_labels)
}

# Call the mclust_R function with the data
result <- mclust_R(data,num_cluster=8, modelNames = "EEE", used_obsm = "MY_ARCH", random_seed = 2020) # nolint
result <- unname(result)

# Write the result to a file
write.csv(result, "C:\\Users\\KARAN\\Desktop\\MultiOmics-Research\\arch2\\protein\\results.csv", row.names = FALSE) # nolint

