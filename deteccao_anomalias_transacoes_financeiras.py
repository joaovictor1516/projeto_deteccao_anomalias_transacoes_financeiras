import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

#Class = 0 -> nao e fraudulenta
#Class = 1 -> e fraudulenta

proporcoes = df["Class"].value_counts(normalize=True)
print(proporcoes)

#feature engeneering e o processo de criar novas variaveis, ou normalizando variaveis ja exitentes, para ajudar no estudo da nossa ia

df["Amount_log"] = np.log1p(df["Amount"]) #usamos o log para diminuir os valores da coluna

df["Amount_scaled"] = scaler.fir_transform(df["Amount"])