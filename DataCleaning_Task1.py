import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset with the correct delimiter
data_frame = pd.read_csv('bank-additional-full (3).csv', delimiter=';')

# Create a copy of the original dataset
cleaned_data_frame = data_frame.copy()

# Check initial dataset structure
print("Initial dataset info:")
print(cleaned_data_frame.info())
print("Initial dataset Head:")
print(cleaned_data_frame.head())
print("Initial dataset Summary:")
print(cleaned_data_frame.describe())

# Check for missing values ('unknown' for this dataset)
print(cleaned_data_frame.isin(['unknown']).sum())

# Boxplot for outlier detection in 'age'
sns.boxplot(x=cleaned_data_frame['age'])
plt.show()

# Change 'unknown' in 'default' to 'no' as most people don't default
cleaned_data_frame['default'] = cleaned_data_frame['default'].replace('unknown', 'no')

# Impute 'unknown' values with the mode for 'housing' and 'loan'
cleaned_data_frame['housing'] = cleaned_data_frame['housing'].replace('unknown', cleaned_data_frame['housing'].mode()[0])
cleaned_data_frame['loan'] = cleaned_data_frame['loan'].replace('unknown', cleaned_data_frame['loan'].mode()[0])

# One-Hot Encoding categorical variables
cleaned_data_frame = pd.get_dummies(cleaned_data_frame, columns=['job', 'marital', 'default', 'housing',
                                                                 'loan', 'month', 'day_of_week', 'poutcome',
                                                                 'education', 'contact'])

# Drop 'duration' as it's not usable for pre-call predictions
cleaned_data_frame.drop(columns=['duration'], inplace=True)

# Numerical Columns
number_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
               'cons.conf.idx', 'euribor3m', 'nr.employed']

# Initialise the scaler
scaler = StandardScaler()

# Scale the numerical columns
cleaned_data_frame[number_cols] = scaler.fit_transform(cleaned_data_frame[number_cols])

# Save the cleaned data to a new CSV file (added .csv extension)
cleaned_data_frame.to_csv('Cleaned-Bank_additional_full.csv', index=False)
