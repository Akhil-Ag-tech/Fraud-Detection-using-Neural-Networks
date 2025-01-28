import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings as wr
wr.filterwarnings(action="ignore")
import seaborn as sns
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive
account = pd.read_csv("account_activity.csv")
customer = pd.read_csv("customer_data.csv")
fraud = pd.read_csv("fraud_indicators.csv")
suspision = pd.read_csv("suspicious_activity.csv")
merchant = pd.read_csv("merchant_data.csv")
tran_cat = pd.read_csv("transaction_category_labels.csv")
amount = pd.read_csv("amount_data.csv")
anamoly = pd.read_csv("anomaly_scores.csv")
tran_data = pd.read_csv("transaction_metadata.csv")
tran_rec = pd.read_csv("transaction_records.csv")
data = [account,customer,fraud,suspision,merchant,tran_cat,amount,anamoly,tran_data,tran_rec]
for df in data:
    print(df.head())
costumer_data = pd.merge(customer, account, on='CustomerID')
costumer_data = pd.merge(costumer_data, suspision, on='CustomerID')
costumer_data
transaction_data1 = pd.merge(fraud, tran_cat, on="TransactionID")
transaction_data2 = pd.merge(amount, anamoly, on="TransactionID")
transaction_data3 = pd.merge(tran_data, tran_rec, on="TransactionID")
transaction_data = pd.merge(transaction_data1, transaction_data2,on="TransactionID")
transaction_data = pd.merge(transaction_data, transaction_data3,on="TransactionID")
data = pd.merge(transaction_data, costumer_data,on="CustomerID")
data
data.info()
data.shape
data.describe()
data.columns
numerical_features = data.select_dtypes(include=['number']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
print(numerical_features)
print(categorical_features)
for column in data.columns:
    if data[column].dtype == 'object':  # Check if the column has a categorical data type
        top_10_values = data[column].value_counts().head(10)  # Get the first 10 unique values and their counts
        plt.figure(figsize=(10, 5))  # Adjust the figure size if needed
        sns.countplot(x=column, data=data, order=top_10_values.index)
        plt.title(f'Count Plot for {column}')
        plt.xticks(rotation=90)  # Rotate x-axis labels if they are long
        plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame containing numerical columns

# Get the number of numerical columns
num_cols = len(data.select_dtypes(include=['number']).columns)

# Calculate the number of rows and columns for subplots
num_rows = (num_cols // 2) + (num_cols % 2)

# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6*num_rows))
fig.suptitle("Box Plots for Numerical Columns")

# Loop through the numerical columns and create box plots
for i, column in enumerate(data.select_dtypes(include=['number']).columns):
    row = i // 2
    col = i % 2
    sns.boxplot(x=data[column], ax=axes[row, col])
    axes[row, col].set_title(column)

# Remove any empty subplots
if num_cols % 2 != 0:
    fig.delaxes(axes[num_rows-1, 1])

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust the position of the overall title
plt.show()
# We should use countplot for SuspiciousFlag feature

plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.countplot(x='SuspiciousFlag', data=data, palette='Set2')  # You can change the palette as desired
plt.title('Count Plot for Suspicious Flag')
plt.xlabel('Suspicious Flag')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels if they are long

plt.show()
# Select only the numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns
correlation_matrix = numeric_data.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numeric Columns')

plt.show()
# Dropping the columns as of now they are not mush corelated & also wouldn't damper the performance of model

columns_to_be_dropped = ['TransactionID','MerchantID','CustomerID','Name', 'Age', 'Address']
data1 = data.drop(columns_to_be_dropped, axis=1)
data1.head()
data1['FraudIndicator'].value_counts(), data1['SuspiciousFlag'].value_counts(), data1['Category'].value_counts()
# Using Feature Engineering Creating two Columns
# Hour of Transaction = hour
# Gap between the day of transaction and last login in days = gap
if pd.api.types.is_datetime64_any_dtype(data['Timestamp']):
    print("The 'Timestamp' column is already in datetime format.")
else:
    print("The 'Timestamp' column is not in datetime format.")
data1['Timestamp1'] = pd.to_datetime(data1['Timestamp'])

print(data1.dtypes)
data1['Hour'] = data1['Timestamp1'].dt.hour
data1['LastLogin'] = pd.to_datetime(data1['LastLogin'])
data1['gap'] = (data1['Timestamp1'] - data1['LastLogin']).dt.days.abs()
data1.head()
X = data1.drop(['FraudIndicator','Timestamp','Timestamp1','LastLogin'],axis=1)
Y = data1['FraudIndicator']
from sklearn.preprocessing import LabelEncoder
# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Category' column
X['Category'] = label_encoder.fit_transform(X['Category'])
X
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
X_train.shape,Y_test.shape
# Logistic Regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, Y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'df' with a 'FraudIndicator' column
# Load your data into the DataFrame if not already done

# Create a count plot for the 'FraudIndicator' column
plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
sns.countplot(data=data1, x='FraudIndicator', palette='viridis')
plt.title('Count Plot of Fraud Indicator')
plt.xlabel('Fraud Indicator')
plt.ylabel('Count')
plt.show()
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features to [0, 1] range
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for oversampling
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Define FROST function
def generate_frost_samples(X_minority, initial_feature_index, k=5, m=1.5):
    initial_feature_values = X_minority[:, initial_feature_index]
    similarity_matrix = 1 / (1 + np.abs(initial_feature_values[:, np.newaxis] - initial_feature_values))
    k_nearest_indices = np.argsort(similarity_matrix, axis=1)[:, -k:]
    synthetic_samples_initial = []
    for i in range(len(initial_feature_values)):
        for j in k_nearest_indices[i]:
            synthetic_value = initial_feature_values[i] + m * (initial_feature_values[j] - initial_feature_values[i])
            synthetic_sample = np.copy(X_minority[i])
            synthetic_sample[initial_feature_index] = synthetic_value
            synthetic_samples_initial.append(synthetic_sample)
    return np.array(synthetic_samples_initial)

# Apply FROST for oversampling
initial_feature_index = 0  # Choose the index of the initial feature to oversample
X_train_frost = generate_frost_samples(X_train_scaled[y_train == 1], initial_feature_index, k=5, m=1.5)

# Combine original and synthetic samples
X_train_combined = np.vstack((X_train_scaled, X_train_frost))
y_train_combined = np.concatenate((y_train, np.ones(len(X_train_frost))))

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Define the number of folds for k-fold cross-validation
k_folds = KFold(n_splits=5)

# Perform cross-validation and calculate the scores for SMOTE
scores_smote = cross_val_score(clf, X_train_smote, y_train_smote, cv=k_folds)

# Perform cross-validation and calculate the scores for FROST
scores_frost = cross_val_score(clf, X_train_combined, y_train_combined, cv=k_folds)

# Print the cross-validation scores for each fold
print("SMOTE Cross Validation Scores: ", scores_smote)
print("FROST Cross Validation Scores: ", scores_frost)

# Print the average cross-validation score
print("Average SMOTE CV Score: ", scores_smote.mean())
print("Average FROST CV Score: ", scores_frost.mean())
# Retraining Logistic regression using SAMPLED Data

model = LogisticRegression()

# Train the model on the training data
model.fit(X_train_smote, y_train_smote)
model.fit(X_train_combined, y_train_combined)


# Make predictions on the testing data
y_predSMOTE = model.predict(X_test)
y_predFROST = model.predict(X_test)

# Calculate and print various metrics to evaluate the model's performance
accuracySMOTE = accuracy_score(Y_test, y_predSMOTE)
precisionSMOTE = precision_score(Y_test, y_predSMOTE)
recallSMOTE = recall_score(Y_test, y_predSMOTE)
f1SMOTE = f1_score(Y_test, y_predSMOTE)
confusionSMOTE = confusion_matrix(Y_test, y_predSMOTE)

accuracyFROST = accuracy_score(Y_test, y_predFROST)
precisionFROST = precision_score(Y_test, y_predFROST)
recallFROST = recall_score(Y_test, y_predFROST)
f1FROST = f1_score(Y_test, y_predFROST)
confusionFROST = confusion_matrix(Y_test, y_predFROST)

print("Model Evaluation Metrics: SMOTE")
print("Accuracy:", accuracySMOTE)
print("Precision:", precisionSMOTE)
print("Recall:", recallSMOTE)
print("F1 Score:", f1SMOTE)
print("Confusion Matrix:")
print(confusionSMOTE)

print("Model Evaluation Metrics: FROST")
print("Accuracy:", accuracyFROST)
print("Precision:", precisionFROST)
print("Recall:", recallFROST)
print("F1 Score:", f1FROST)
print("Confusion Matrix:")
print(confusionFROST)
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the logistic regression model
model = LogisticRegression()

# Define a range of hyperparameters to search
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization type
    'C': np.logspace(-3, 3, 7),  # Inverse of regularization strength (smaller values for stronger regularization)
    'solver': ['liblinear'],  # Solver for l1 regularization
}

# Create a grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train_smote, y_train_smote)

# Get the best hyperparameters and corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Evaluate the best model on the resampled data
y_pred = best_model.predict(X_train_smote)

# Calculate and print various metrics to evaluate the model's performance on the resampled data
accuracy = accuracy_score(y_train_smote, y_pred)
precision = precision_score(y_train_smote, y_pred)
recall = recall_score(y_train_smote, y_pred)
f1 = f1_score(y_train_smote, y_pred)
confusion = confusion_matrix(y_train_smote, y_pred)

print("Model Evaluation Metrics on Resampled Data- SMOTE:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the logistic regression model
model = LogisticRegression()

# Define a range of hyperparameters to search
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization type
    'C': np.logspace(-3, 3, 7),  # Inverse of regularization strength (smaller values for stronger regularization)
    'solver': ['liblinear'],  # Solver for l1 regularization
}

# Create a grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train_combined, y_train_combined)

# Get the best hyperparameters and corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Evaluate the best model on the resampled data
y_pred = best_model.predict(X_train_combined)

# Calculate and print various metrics to evaluate the model's performance on the resampled data
accuracy = accuracy_score(y_train_combined, y_pred)
precision = precision_score(y_train_combined, y_pred)
recall = recall_score(y_train_combined, y_pred)
f1 = f1_score(y_train_combined, y_pred)
confusion = confusion_matrix(y_train_combined, y_pred)

print("Model Evaluation Metrics on Resampled Data- FROST:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def evaluate_classification_models(X_train_smote, y_train_smote):
    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_smote, y_train_smote, test_size=0.2, random_state=42)

    # Define a dictionary of classification models
    models = {
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Support Vector Machine (SVM)": SVC(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier()
    }

    results = {}

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and store various metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": confusion
        }

    return results

results = evaluate_classification_models(X_train_smote, y_train_smote)
for model_name, model_result in results.items():
     print(f"Results for {model_name}:")
     for metric, value in model_result.items():
         print(f"{metric}: {value}")
     print()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def evaluate_classification_models(X_train_combined, y_train_combined):
    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_combined, y_train_combined, test_size=0.2, random_state=42)

    # Define a dictionary of classification models
    models = {
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Support Vector Machine (SVM)": SVC(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier()
    }

    results = {}

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and store various metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": confusion
        }

    return results

results = evaluate_classification_models(X_train_combined, y_train_combined)
for model_name, model_result in results.items():
     print(f"Results for {model_name}:")
     for metric, value in model_result.items():
         print(f"{metric}: {value}")
     print()
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_combined, y_train_combined, test_size=0.2, random_state=42)

# Define the Random Forest Classifier model
rf_model = RandomForestClassifier(random_state=42)

# Define a range of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
}

# Create a grid search with cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1)

# Fit the grid search to the resampled data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and corresponding model
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Train the best model on the training data
best_rf_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = best_rf_model.predict(X_test)

# Calculate and print various metrics to evaluate the best model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Best Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)

import datetime

def check_transaction(account_open_date, transaction_date, amount):
  """
  Checks if a transaction is suspicious based on account open date, transaction date, and amount.

  Args:
    account_open_date: The date the account was opened.
    transaction_date: The date of the transaction.
    amount: The amount of the transaction.

  Returns:
    True if the transaction is suspicious, False otherwise.
  """

  # Calculate the time difference between account open date and transaction date.
  time_difference = transaction_date - account_open_date

  # Check if the transaction occurs within 24 hours of account creation.
  if time_difference.days < 1:
    return True

  # Check if the transaction amount is greater than 1000000.
  if amount > 1000000:
    return True

  # Otherwise, the transaction is not suspicious.
  return False

# Get user input for account open date, transaction date, and amount.
account_open_date_str = input("Enter the account open date (YYYY-MM-DD): ")
transaction_date_str = input("Enter the transaction date (YYYY-MM-DD): ")
amount = float(input("Enter the transaction amount: "))

# Convert strings to datetime objects.
account_open_date = datetime.datetime.strptime(account_open_date_str, "%Y-%m-%d")
transaction_date = datetime.datetime.strptime(transaction_date_str, "%Y-%m-%d")

# Check if the transaction is suspicious.
is_suspicious = check_transaction(account_open_date, transaction_date, amount)

# Print the result.
if is_suspicious:
  print("The transaction is suspicious.")
else:
  print("The transaction is not suspicious.")

import datetime

def check_suspicious_activity(timestamp):
  """
  Checks if a timestamp indicates suspicious activity.

  Args:
    timestamp: A datetime object representing the timestamp to check.

  Returns:
    True if the timestamp is suspicious, False otherwise.
  """

  # Get the current date.
  today = datetime.datetime.now()

  # Check if the timestamp is after today's date.
  if timestamp > today:
    print("Invalid timestamp: Timestamp is after today's date.")
    return False

  # Calculate the time difference between the timestamp and today.
  time_difference = today - timestamp

  # Check if the time difference is greater than 6 months.
  if time_difference > datetime.timedelta(days=180):
    print("Suspicious activity: Long period of inactivity followed by sudden transaction.")
    return True

  # Otherwise, the timestamp is not suspicious.
  return False

# Get user input for the timestamp.
timestamp_str = input("Enter the timestamp (YYYY-MM-DD HH:MM:SS): ")

# Parse the timestamp string into a datetime object.
timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

# Check for suspicious activity.
if check_suspicious_activity(timestamp):
  print("The timestamp indicates suspicious activity.")
else:
  print("The timestamp does not indicate suspicious activity.")

def check_transaction(account_balance, transaction_amount):
  """
  Checks if a transaction is suspicious based on the customer's account balance.

  Args:
    account_balance: The customer's account balance.
    transaction_amount: The amount of the transaction.

  Returns:
    True if the transaction is suspicious, False otherwise.
  """

  # Calculate the threshold for suspicious transactions.
  threshold = 0.8 * account_balance

  # Check if the transaction amount exceeds the threshold.
  if transaction_amount > threshold:
    return True
  else:
    return False

# Get user input for account balance and transaction amount.
account_balance = float(input("Enter the customer's account balance: "))
transaction_amount = float(input("Enter the transaction amount: "))

# Check if the transaction is suspicious.
is_suspicious = check_transaction(account_balance, transaction_amount)

# Print the result.
if is_suspicious:
  print("The transaction is suspicious.")
else:
  print("The transaction is not suspicious.")

def set_category_limits():
  categories = {
      "Electronics": 50000,
      "Travel": 200000,
      "Food": 10000,
      "Entertainment": 5000,
      "Clothing": 20000,
      "Health": 15000,
      "Transportation": 10000,
      "Utilities": 5000,
      "Other": 10000
  }
  return categories

def check_transaction(category, amount):
  limits = set_category_limits()
  if category in limits:
    if amount <= limits[category]:
      return True
    else:
      return False
  else:
    return False

# Example usage
category = input("Enter transaction category: ")
amount = int(input("Enter transaction amount: "))

if check_transaction(category, amount):
  print("Transaction allowed.")
else:
  print("Transaction declined. Amount exceeds limit for category.")
category = input("Enter transaction category: ")
amount = int(input("Enter transaction amount: "))

if check_transaction(category, amount):
  print("Transaction allowed.")
else:
  print("Transaction declined. Amount exceeds limit for category.")

def fraud_detection(age, bank_balance, category, transaction_amount):
  if age < 18:
    if transaction_amount > 10000:
      if category == "luxury":
        return "Fraudulent transaction"
      else:
        return "Suspicious transaction"
    else:
      return "Normal transaction"
  else:
    if transaction_amount > bank_balance:
      return "Fraudulent transaction"
    else:
      return "Normal transaction"

# Example usage
age = int(input("Enter your age: "))
bank_balance = int(input("Enter your bank balance: "))
category = input("Enter the transaction category: ")
transaction_amount = int(input("Enter the transaction amount: "))

result = fraud_detection(age, bank_balance, category, transaction_amount)
print(result)

