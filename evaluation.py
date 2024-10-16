import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# Load the feature-engineered dataset to use for model evaluation
data = pd.read_csv('FeatureEngineered_BankData.csv')

# Separate the target variable 'y' from the features. Convert 'y' to binary (1 if 'yes', 0 if 'no')
X = data.drop(columns=['y'])
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Load the pre-trained Logistic Regression and Random Forest models from disk
logreg_best = joblib.load('logreg_best_model.pkl')
rf_best = joblib.load('rf_best_model.pkl')

# --- Logistic Regression Evaluation ---
# Generate predictions and predicted probabilities for Logistic Regression
logreg_preds = logreg_best.predict(X)
logreg_probs = logreg_best.predict_proba(X)[:, 1]  # Extract probabilities for the positive class (1)

print("\n=== Logistic Regression Evaluation ===")
# Calculate and display various evaluation metrics for Logistic Regression
print(f"Accuracy: {accuracy_score(y, logreg_preds):.4f}")  # How often the model is correct
print(f"Precision: {precision_score(y, logreg_preds):.4f}")  # Precision: true positives / (true positives + false positives)
print(f"Recall: {recall_score(y, logreg_preds):.4f}")  # Recall: true positives / (true positives + false negatives)
print(f"F1-Score: {f1_score(y, logreg_preds):.4f}")  # F1 Score: harmonic mean of precision and recall
print(f"AUC-ROC: {roc_auc_score(y, logreg_probs):.4f}")  # Area under the ROC curve to measure classification performance

# --- Random Forest Evaluation ---
# Generate predictions and predicted probabilities for Random Forest
rf_preds = rf_best.predict(X)
rf_probs = rf_best.predict_proba(X)[:, 1]  # Extract probabilities for the positive class (1)

print("\n=== Random Forest Evaluation ===")
# Calculate and display various evaluation metrics for Random Forest
print(f"Accuracy: {accuracy_score(y, rf_preds):.4f}")  # How often the model is correct
print(f"Precision: {precision_score(y, rf_preds):.4f}")  # Precision: true positives / (true positives + false positives)
print(f"Recall: {recall_score(y, rf_preds):.4f}")  # Recall: true positives / (true positives + false negatives)
print(f"F1-Score: {f1_score(y, rf_preds):.4f}")  # F1 Score: harmonic mean of precision and recall
print(f"AUC-ROC: {roc_auc_score(y, rf_probs):.4f}")  # Area under the ROC curve to measure classification performance

# Generate and display the confusion matrix for Logistic Regression
conf_matrix = confusion_matrix(y, logreg_preds)
print("\nConfusion Matrix for Logistic Regression:")
print(conf_matrix)  # Confusion matrix: a summary of prediction results by showing true vs. predicted classes
