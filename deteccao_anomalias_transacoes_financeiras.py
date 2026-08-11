import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

proporcoes = df["Class"].value_counts(normalize=True)
print(proporcoes)

df["Amount_log"] = np.log1p(df["Amount"])

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)