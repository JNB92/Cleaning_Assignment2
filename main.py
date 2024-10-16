import os

# Step 1: Data cleaning and preparation
os.system("python DataCleaning_Task1.py")

# Step 2: Feature engineering
os.system("python feature_engineering.py")

# Step 3: Train models
os.system("python modeling.py")

# Step 4: Evaluate models
os.system("python evaluation.py")