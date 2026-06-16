# ==========================================
# File 7: hyperparameter_tuning.py
# Purpose: Tune hyperparameters of Random Forest using GridSearchCV.
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Import split_dataset from File 5 (train_test_split_file)
from train_test_split_file import split_dataset

def tune_random_forest():
    """
    Function to optimize the Random Forest Classifier.
    Uses GridSearchCV (Grid Search Cross-Validation) to find the best:
    - n_estimators (number of decision trees in the forest)
    - max_depth (maximum depth of each decision tree)
    
    GridSearchCV runs cross-validation for all parameter combinations and identifies
    the combination that yields the highest average score.
    """
    # 1. Load split dataset
    X_train, X_test, y_train, y_test = split_dataset()
    
    print("\n--- Hyperparameter Tuning using GridSearchCV ---")
    
    # 2. Set up parameter grid to search
    # n_estimators: [50, 100] (fewer trees vs default trees)
    # max_depth: [5, 10, None] (shallow trees, medium depth, or fully grown trees)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }
    
    # 3. Initialize base Random Forest model
    rf = RandomForestClassifier(random_state=42)
    
    # 4. Initialize GridSearchCV
    # cv=3 means 3-fold cross-validation
    # verbose=1 shows basic progress logs
    # n_jobs=-1 uses all available processor cores
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # 5. Fit GridSearchCV on the training data
    print("Starting Grid Search fit... This will test 6 combinations (2 n_estimators * 3 max_depths) using 3-fold CV...")
    grid_search.fit(X_train, y_train)
    
    # 6. Retrieve best parameters and the best score achieved
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("\n--- Tuning Results ---")
    print(f"Best Parameters Found: {best_params}")
    print(f"Best CV Accuracy Score:  {best_score * 100:.2f}%")
    
    # 7. Evaluate the best tuned model on testing data
    best_rf = grid_search.best_estimator_
    test_acc = best_rf.score(X_test, y_test)
    print(f"Accuracy of Tuned Model on Test Set: {test_acc * 100:.2f}%")
    
    return best_rf, best_params

if __name__ == "__main__":
    tune_random_forest()
