import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

proporcoes = df["Class"].value_counts(normalize=True)
print(proporcoes)

df["Amount_log"] = np.log1p(df["Amount"])

df["Amount_scaled"] = scaler.fir_transform(df["Amount"])