import pandas as pd


# Load the dataset with the correct delimiter
data_frame = pd.read_csv('bank-additional-full.csv', delimiter=';')

# Check initial dataset structure
print("Initial dataset info:")
print(data_frame.info())

# Replace any value marked as unknown with NA
data_frame.replace('unknown', pd.NA, inplace=True)




