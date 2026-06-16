# ==========================================
# File 5: train_test_split_file.py
# Purpose: Split the preprocessed dataset into train and test subsets.
# ==========================================

from sklearn.model_selection import train_test_split
# Import preprocess_data from File 3 (data_preprocessing)
from data_preprocessing import preprocess_data

def split_dataset():
    """
    Function to split features (X) and target (y) into:
    - 80% Training data (used to train the models)
    - 20% Testing data (used to evaluate performance)
    Uses random_state=42 for reproducibility.
    """
    # 1. Get features and target from preprocessing
    X, y = preprocess_data()
    
    print("\n--- Splitting Dataset into Train and Test Sets ---")
    
    # 2. Perform train/test split
    # test_size=0.2 represents 20% test data, 80% training data
    # random_state=42 ensures we get the same split every time we run the script
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Print shapes of the split subsets
    print(f"Training Features shape (X_train): {X_train.shape}")
    print(f"Testing Features shape (X_test):   {X_test.shape}")
    print(f"Training Labels shape (y_train):   {y_train.shape}")
    print(f"Testing Labels shape (y_test):     {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_dataset()
    print("\nTrain-test split completed successfully!")
