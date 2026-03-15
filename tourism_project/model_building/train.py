# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Set tracking url for mlops
mlflow.set_tracking_uri("http://localhost:5000")

# Set the name for the experiment
mlflow.set_experiment("mlops-tourism-training-experiment")

# Set HuggingFace API token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load train and test data wih force download to avoid cache issues
Xtrain_path = "hf://datasets/navzen2000/tourism_project/Xtrain.csv"
Xtest_path = "hf://datasets/navzen2000/tourism_project/Xtest.csv"
ytrain_path = "hf://datasets/navzen2000/tourism_project/ytrain.csv"
ytest_path = "hf://datasets/navzen2000/tourism_project/ytest.csv"

X_train = pd.read_csv(Xtrain_path, storage_options={"force_download": True})
X_test = pd.read_csv(Xtest_path, storage_options={"force_download": True})
y_train = pd.read_csv(ytrain_path, storage_options={"force_download": True})
y_test = pd.read_csv(ytest_path, storage_options={"force_download": True})


# List of numerical features in the dataset
numeric_features = [
    'Age',                        # Age of the customer
    'DurationOfPitch',            # Duration of the sales pitch delivered to the customer
    'NumberOfFollowups',          # Total number of follow-ups by the salesperson after the sales pitch
    'NumberOfTrips',              # Average number of trips the customer takes annually
    'MonthlyIncome',              # Gross monthly income of the customer
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',              # The method by which the customer was contacted 
    'CityTier',                   # The city category based on development, population, and living standards
    'Occupation',                 # Customer's occupation
    'Gender',                     # Gender of the customer
    'NumberOfPersonVisiting',     # Total number of people accompanying the customer on the trip
    'ProductPitched',             # The type of product pitched to the customer
    'PreferredPropertyStar',      # Preferred hotel rating by the customer
    'MaritalStatus',              # Marital status of the customer
    'Passport',                   # Whether the customer holds a valid passport
    'PitchSatisfactionScore',     # Score indicating the customer's satisfaction with the sales pitch
    'OwnCar',                     # Whether the customer owns a car
    'NumberOfChildrenVisiting',   # Number of children below age 5 accompanying the customer
    'Designation'                 # Customer's designation in their current organization
]

# Create Preprocessor with Standard Scaling of numeric columns and OneHotEncoding for Categorical Columns
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
)

# Calculate class weight for imbalance
class_weight = y_train.value_counts().iloc[0] / y_train.value_counts().iloc[1]

## Define base XGBoost model with below attributes

# XGBClassifier: This initializes the model
# use_label_encoder=False: This stops XGBoost from using its own internal label encoder
# eval_metric='logloss': This tells the model to monitor Logarithmic Loss during training
# scale_pos_weight=class_weight: It tells the model to pay more attention to the minority class. 
# random_state=42: This ensures that every time you run the code, you get the same results
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',scale_pos_weight=class_weight, random_state=42)

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Define Hyperparameter Tuning Parameters
param_grid = {
    'xgbclassifier__n_estimators': [100, 150, 200],           # number of tree to build
    'xgbclassifier__max_depth': [2, 3],                       # maximum depth of each tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],        # learning rate
    'xgbclassifier__colsample_bytree': [0.5, 0.6],            # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__subsample': [0.7, 0.8],                   # row sampling for robustness
    'xgbclassifier__reg_lambda': [1, 5, 10],                  # L2 regularization factor
    'xgbclassifier__gamma': [0.1, 0.2, 0.5],                  # Minimum loss reduction to create a new node (pruning)
    'xgbclassifier__min_child_weight': [3, 5, 7]              # prevents over-specific leaves
}

# Tune Model & Log Parameters (MLflow) ---
with mlflow.start_run(run_name="XGB_Tuning_Session"):
    # Initialize a cross-validated grid search to find the optimal hyperparameters
    grid_search = GridSearchCV(
        estimator=model_pipeline,  # The machine learning pipeline or model to be evaluated
        param_grid=param_grid,     # Dictionary containing the hyperparameter values to test
        cv=5,                      # 5-fold cross-validation strategy to ensure model generalization
        n_jobs=-1,                 # Run jobs in parallel utilizing all CPU cores
        scoring='f1',              # Evaluate performance using the F1-score to balance precision and recall
        verbose=3                  # Verbosity Level
    )
    # Fit the model
    grid_search.fit(X_train, y_train)

   
# Log all parameter combinations as nested runs

results = grid_search.cv_results_
for i in range(len(results['params'])):
    param_set = results['params'][i]
    mean_score = results['mean_test_score'][i]
    std_score = results['std_test_score'][i]

    # Log each combination as a separate MLflow run
    with mlflow.start_run(nested=True):
        mlflow.log_params(param_set)
        mlflow.log_metric("mean_test_score", mean_score)
        mlflow.log_metric("std_test_score", std_score)


# Log best parameters separately in main run
mlflow.log_params(grid_search.best_params_)

# Store and evaluate the best model
best_model = grid_search.best_estimator_

print("Best Params: \n",grid_search.best_params_)

#  Evaluate Performance ---
classification_threshold = 0.45 # Set threshold of 0.45
y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

train_report = classification_report(y_train, y_pred_train, output_dict=True)
test_report = classification_report(y_test, y_pred_test, output_dict=True)

print("\nTrain Metrics: \n",classification_report(y_train, y_pred_train))
print("\nTest Metrics: \n",classification_report(y_test, y_pred_test))

mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
})

# Register Best Model in Hugging Face Hub ---

# Save the model locally
model_path = "best_tourism_model_prod.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact
mlflow.log_artifact(model_path,  artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = "navzen2000/tourism-model"
repo_type = "model"


#Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="best_tourism_model_prod.joblib",
    path_in_repo="best_tourism_model_prod.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"✅ Success! Model uploaded to: https://huggingface.co/navzen2000/tourism-model")
