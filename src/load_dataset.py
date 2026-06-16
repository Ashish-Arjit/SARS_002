# ==========================================
# File 2: load_dataset.py
# Purpose: Load the crop recommendation dataset and display its basic details.
# ==========================================

import os
import pandas as pd

def load_data():
    """
    Function to load the Crop Recommendation dataset.
    Uses dynamic path resolution so that it can be run from any working directory.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the dataset folder (parent directory, then dataset/)
    dataset_path = os.path.join(current_dir, "..", "dataset", "Crop_recommendation.csv")
    
    # Check if the dataset file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}. Please make sure to download the dataset first.")
    
    # Load the CSV file using Pandas
    df = pd.read_csv(dataset_path)
    return df

if __name__ == "__main__":
    print("--- Loading Dataset ---")
    
    # Load the dataset
    data = load_data()
    
    # 1. Display the shape (number of rows and columns) of the dataset
    print(f"\nDataset Shape (Rows, Columns): {data.shape}")
    
    # 2. Display the column names (features and target label)
    print("\nColumns in the Dataset:")
    print(list(data.columns))
    
    # 3. Display the first 5 rows of the dataset
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    print("\nDataset loaded successfully!")
