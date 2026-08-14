import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, roc_curve, roc_auc_score

scaler = StandardScaler()

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

df["Amount_log"] = np.log1p(df["Amount"])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)

x_train["Amount_scaled"] = scaler.fit_transform(x_train[["Amount"]])
x_test["Amount_scaled"] = scaler.fit_transform(x_test[["Amount"]])

train = pd.concat([x_train, y_train], axis=1)

fraudes = train[train["Class"] == 1]
normais = train[train["Class"] == 0].sample(len(fraudes), random_state=42)

train_under = pd.concat([fraudes, normais])

x_train_under = train_under.drop("Class", axis=1)
y_train_under = train_under["Class"]

# Logistic Regression:
model = LogisticRegression(max_iter=10000)

model.fit(x_train_under, y_train_under)

y_predict = model.predict(x_test)

print(classification_report(y_test, y_predict))

y_probs = model.predict_proba(x_test)[:,1]

ftp, rtp, _ = roc_curve(y_test, y_probs)

plt.plot(ftp, rtp)
plt.title("Roc Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC: ", roc_auc_score(y_test, y_probs))

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision X Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Random Forest:
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=1,
    random_state=42
)

rf.fit(x_train_under, y_train_under)

y_predict_rf = rf.predict(x_test)

print(classification_report(y_test, y_predict_rf))

y_probs_rf = rf.predict_proba(x_test)[:,1]

print("AUC:", roc_auc_score(y_test, y_probs_rf))