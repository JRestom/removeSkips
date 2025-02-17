#!/bin/bash

# Job name
#SBATCH --job-name=cifar10_batched
# Compute resources
#SBATCH --ntasks=1                  # Number of tasks per node
#SBATCH --cpus-per-task=1           # CPU cores per task (adjust based on dataset loading requirements)
#SBATCH --gpus=1                    # Number of GPUs to use
#SBATCH --mem=40G                   # Total memory required
#SBATCH --time=24:00:00             # Maximum time allowed
#SBATCH -p cscc-gpu-p               # GPU partition
#SBATCH -q cscc-gpu-qos             # GPU queue

# Load necessary modules
module load anaconda3               # Load Anaconda (adjust for your cluster setup)

# Activate the Fastenet environment
source activate fasternet

# Folder containing the configuration files
CONFIG_FOLDER=$1

# Ensure the folder exists
if [ ! -d "$CONFIG_FOLDER" ]; then
  echo "Error: Configuration folder '$CONFIG_FOLDER' does not exist."
  exit 1
fi

# Iterate through all YAML files in the folder
for CONFIG_FILE in "$CONFIG_FOLDER"/*.yml; do
  if [ -f "$CONFIG_FILE" ]; then
    echo "Running training with configuration: $CONFIG_FILE"
    python cifar_train.py --config "$CONFIG_FILE"
  else
    echo "No configuration files found in the folder."
    exit 1
  fi
done
