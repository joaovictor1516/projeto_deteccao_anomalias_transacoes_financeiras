import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, roc_curve, roc_auc_score

scaler = StandardScaler()
smote = SMOTE()

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

df["Amount_log"] = np.log1p(df["Amount"])

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

x = df.drop("Class", axis=1)
y = df["Class"]

x_res, y_res = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x_res, y_res, stratify=y_res, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=10000)

model.fit(x_train, y_train)

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