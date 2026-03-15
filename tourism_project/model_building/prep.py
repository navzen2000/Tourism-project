# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/navzen2000/tourism_project/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop Unnamed: 0 (index column) and CustomerID
df.drop(columns=['Unnamed: 0','CustomerID'], inplace=True)

# Fix Gender typo ('Fe male' -> 'Female')
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Group MaritalStatus into Single, Married, and Divorced 
# This converts Unmarried to Single
df['MaritalStatus'] = df['MaritalStatus'].apply(lambda x: 'Single' if x == 'Unmarried' else x)

# List of numerical columns that are actually categorical
numerical_categorical_cols = [
    'CityTier', 'NumberOfPersonVisiting', 
    'NumberOfChildrenVisiting', 'PreferredPropertyStar', 
    'Passport', 'PitchSatisfactionScore', 'OwnCar'
]

# Convert these numerical columns to category type
for col in numerical_categorical_cols:
    df[col] = df[col].astype('category')

# Convert remaining object columns to category type
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].astype('category')

# Verify the changes
print(df.info())
print("\nUnique MaritalStatus:", df['MaritalStatus'].unique())
print("Unique Gender:", df['Gender'].unique())



# Define predictor matrix (X) by dropping the target variable
X = df.drop('ProdTaken',axis=1)

# Define target variable
y = df['ProdTaken'].astype(int) # Ensure target is integer for XGB


# Split dataset into train and test
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,    # Ensures reproducibility by setting a fixed random seed
    stratify=y          # Ensures same set of class labels for train and test sets
)

# Print the distributions
print("--- Target Distribution (ProdTaken) ---")
print(f"Training Set:\n{y_train.value_counts(normalize=True).map('{:.2%}'.format)}")
print(f"\nTesting Set:\n{y_test.value_counts(normalize=True).map('{:.2%}'.format)}")

# Print actual counts to see the size
print(f"\nTotal Train samples: {len(y_train)}")
print(f"Total Test samples: {len(y_test)}")

# Save the above split files locally
X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)
