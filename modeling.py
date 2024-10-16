import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Load the dataset that has already been feature-engineered
data = pd.read_csv('FeatureEngineered_BankData.csv')

# Separate the target variable 'y' from the features. Convert 'y' to binary (1 if 'yes', 0 if 'no').
X = data.drop(columns=['y'])
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Step 1: Split the data into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Apply SMOTE to address class imbalance by oversampling the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 3: Train Logistic Regression with class weights to handle imbalance and use cross-validation for evaluation
print("\n=== Logistic Regression with Cross-Validation ===")
logreg = LogisticRegression(max_iter=5000, class_weight='balanced')  # Increase max_iter to ensure convergence
logreg_cv_scores = cross_val_score(logreg, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
print(f"Logistic Regression AUC-ROC (5-fold cross-validation): {logreg_cv_scores}")
print(f"Mean AUC-ROC: {logreg_cv_scores.mean():.4f}")

# Step 4: Train a Random Forest classifier with class weights and evaluate it using cross-validation
print("\n=== Random Forest with Cross-Validation ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_cv_scores = cross_val_score(rf, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
print(f"Random Forest AUC-ROC (5-fold cross-validation): {rf_cv_scores}")
print(f"Mean AUC-ROC: {rf_cv_scores.mean():.4f}")

# Step 5: Perform hyperparameter tuning for Random Forest using RandomizedSearchCV
# This helps find the optimal combination of parameters without exhaustively searching all combinations
print("\n=== Random Forest Hyperparameter Tuning ===")
param_distributions_rf = {
    'n_estimators': [50, 100],      # Test different numbers of trees in the forest
    'max_depth': [10, 15, 20],      # Control the maximum depth of the trees
    'min_samples_split': [5, 10]    # Control the minimum number of samples required to split a node
}

# RandomizedSearchCV tests random combinations of parameters for Random Forest, using AUC-ROC as the scoring metric
random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_distributions_rf, n_iter=5,  # Try 5 random parameter combinations
    cv=3, scoring='roc_auc', random_state=42, n_jobs=2  # Run with 2 parallel jobs to speed up search
)

# Fit RandomizedSearchCV to the training data (SMOTE balanced)
random_search_rf.fit(X_train_smote, y_train_smote)

# Output the best parameters and the best score from the random search
print(f"Best Hyperparameters for Random Forest: {random_search_rf.best_params_}")
print(f"Best AUC-ROC Score (RF): {random_search_rf.best_score_:.4f}")

# Step 6: Train the final Logistic Regression and Random Forest models using the SMOTE-balanced data
logreg.fit(X_train_smote, y_train_smote)
rf_best = random_search_rf.best_estimator_
rf_best.fit(X_train_smote, y_train_smote)

# Step 7: Evaluate both models on the test set using AUC-ROC
logreg_probs = logreg.predict_proba(X_test)[:, 1]
rf_probs = rf_best.predict_proba(X_test)[:, 1]

print(f"\nLogistic Regression AUC-ROC on Test Set: {roc_auc_score(y_test, logreg_probs):.4f}")
print(f"Random Forest AUC-ROC on Test Set: {roc_auc_score(y_test, rf_probs):.4f}")

# Step 8: Evaluate the models' accuracy on the test set to get a broader sense of performance
logreg_preds = logreg.predict(X_test)
rf_preds = rf_best.predict(X_test)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, logreg_preds):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")

# Save the trained models to disk for future use
joblib.dump(logreg, 'logreg_best_model.pkl')
joblib.dump(rf_best, 'rf_best_model.pkl')

print("Models saved as 'logreg_best_model.pkl' and 'rf_best_model.pkl'.")
