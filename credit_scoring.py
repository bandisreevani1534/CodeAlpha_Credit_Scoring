import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# --------------------------------------------------
# Credit Scoring Model - CodeAlpha Internship
# --------------------------------------------------

# Sample financial dataset
data = {
    "age": [25, 45, 35, 50, 23, 40, 29, 55, 31, 48,
            27, 52, 36, 43, 22, 39, 46, 28, 34, 58,
            26, 49, 33, 41, 24, 53, 37, 44, 30, 56],

    "income": [25000, 65000, 45000, 80000, 22000, 55000, 32000, 90000,
               40000, 70000, 28000, 85000, 48000, 60000, 20000, 52000,
               68000, 30000, 42000, 95000, 27000, 75000, 39000, 62000,
               23000, 88000, 50000, 67000, 35000, 92000],

    "loan_amount": [15000, 10000, 20000, 15000, 18000, 12000, 25000, 10000,
                    22000, 15000, 20000, 12000, 18000, 14000, 16000, 13000,
                    11000, 24000, 19000, 10000, 21000, 13000, 23000, 16000,
                    19000, 11000, 17000, 14000, 22000, 9000],

    "existing_loans": [2, 1, 3, 1, 4, 2, 3, 1, 2, 1,
                       4, 1, 2, 2, 5, 2, 1, 4, 3, 1,
                       4, 1, 3, 2, 5, 1, 2, 1, 3, 1],

    # 1 = Good Credit, 0 = Poor Credit
    "credit_score": [0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
                     0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
                     0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

print("======================================")
print("       CREDIT SCORING MODEL")
print("======================================")

print("\nDataset:")
print(df.head())

# Features
X = df[["age", "income", "loan_amount", "existing_loans"]]

# Target
y = df["credit_score"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n======================================")
print("         MODEL PERFORMANCE")
print("======================================")

print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1-Score  : {f1:.2f}")
print(f"ROC-AUC   : {roc_auc:.2f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Poor Credit", "Good Credit"],
    zero_division=0
))

# Example prediction
new_customer = pd.DataFrame({
    "age": [30],
    "income": [50000],
    "loan_amount": [12000],
    "existing_loans": [1]
})

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0][1]

print("\n======================================")
print("       NEW CUSTOMER PREDICTION")
print("======================================")

if prediction == 1:
    print("Credit Status : GOOD CREDIT")
else:
    print("Credit Status : POOR CREDIT")

print(f"Probability of Good Credit: {probability:.2%}")