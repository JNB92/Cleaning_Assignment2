import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset to apply feature engineering techniques
cleaned_data_frame = pd.read_csv('Cleaned-Bank_additional_full.csv')

# Display the current columns in the dataset before feature engineering begins
print("Columns in cleaned dataset before feature engineering:")
print(cleaned_data_frame.columns)

# Feature Engineering Steps:
# 1. Create age groups by binning the 'age' column into categories: 'young', 'middle-aged', and 'old'
cleaned_data_frame['age_group'] = pd.cut(
    cleaned_data_frame['age'], bins=[18, 30, 50, 100], labels=['young', 'middle-aged', 'old']
)

# 2. Remove the original 'age' column after creating the age groups since it's redundant now
cleaned_data_frame.drop(columns=['age'], inplace=True)

# Categorical variables like 'job', 'marital', 'education', etc., have already been one-hot encoded.
# Here, we need to encode 'age_group' as it's a new categorical feature.
categorical_columns = ['age_group']

# Apply one-hot encoding to 'age_group', dropping the first category to avoid multicollinearity
cleaned_data_frame = pd.get_dummies(cleaned_data_frame, columns=categorical_columns, drop_first=True)

# 4. Scaling numerical columns: Prepare the numerical features for modeling by scaling them
# This ensures that all features are on the same scale, which is important for models sensitive to magnitude
numerical_columns = ['campaign', 'pdays', 'previous', 'emp.var.rate',
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Initialize the StandardScaler to standardize the numerical columns (mean = 0, std = 1)
scaler = StandardScaler()

# Apply scaling to the selected numerical columns
cleaned_data_frame[numerical_columns] = scaler.fit_transform(cleaned_data_frame[numerical_columns])

# Save the transformed and feature-engineered dataset to a new CSV file for further use
cleaned_data_frame.to_csv('FeatureEngineered_BankData.csv', index=False)
