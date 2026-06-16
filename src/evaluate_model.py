# ==========================================
# File 8: evaluate_model.py
# Purpose: Evaluate the model using Accuracy, Classification Report, and Confusion Matrix.
# ==========================================

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Import split_dataset from File 5 and tune_random_forest from File 7
from train_test_split_file import split_dataset
from hyperparameter_tuning import tune_random_forest

def evaluate_model():
    """
    Function to evaluate the performance of our tuned Random Forest classifier.
    It prints:
    1. Accuracy Score
    2. Classification Report (Precision, Recall, F1-Score)
    3. Confusion Matrix
    And explains what each of these metrics means.
    """
    # 1. Load split test set
    _, X_test, _, y_test = split_dataset()
    
    # 2. Train and get the best tuned Random Forest model
    best_rf, _ = tune_random_forest()
    
    print("\n--- Model Evaluation ---")
    
    # 3. Predict the labels for the test features
    y_pred = best_rf.predict(X_test)
    
    # ==========================================
    # Metric 1: Accuracy Score
    # Explanation:
    # Accuracy is the ratio of correctly predicted instances to the total instances.
    # Formula: (True Positives + True Negatives) / Total Predictions
    # It represents the overall correctness of the model.
    # ==========================================
    accuracy = accuracy_score(y_test, y_pred)
    print("\n1. [ACCURACY SCORE]")
    print(f"Overall Testing Accuracy: {accuracy * 100:.2f}%")
    print("Meaning: Out of all the test samples, our model recommended the correct crop in this percentage of cases.")
    
    # ==========================================
    # Metric 2: Classification Report
    # Explanation:
    # Provides precision, recall, and f1-score for each crop class.
    # - Precision: Out of all crops predicted as Class A, how many were actually Class A?
    #   (Low precision means high false positives)
    # - Recall (Sensitivity): Out of all actual Class A crops in the dataset, how many did the model identify?
    #   (Low recall means high false negatives)
    # - F1-score: The harmonic mean of Precision and Recall. High F1-score means balanced performance.
    # - Support: The actual number of occurrences of each class in the test set.
    # ==========================================
    report = classification_report(y_test, y_pred)
    print("\n2. [CLASSIFICATION REPORT]")
    print(report)
    
    # ==========================================
    # Metric 3: Confusion Matrix
    # Explanation:
    # A matrix where row indices represent actual classes and column indices represent predicted classes.
    # - Diagonal elements show correct predictions.
    # - Off-diagonal elements show misclassifications (errors).
    # It helps us pinpoint exactly which crops are being confused with each other by the model.
    # ==========================================
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\n3. [CONFUSION MATRIX]")
    print(conf_matrix)
    print("\nMeaning of Confusion Matrix:")
    print("- Rows represent Actual Crops, and Columns represent Predicted Crops.")
    print("- Higher values along the diagonal indicate highly accurate predictions.")
    
    return best_rf, accuracy

if __name__ == "__main__":
    evaluate_model()
