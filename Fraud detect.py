
#essentials
import pandas as pd

df = pd.read_csv('Data/creditcard.csv')
df.head()

df.info()
df['Class'].value_counts()

# Separate the samples by class
legit = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

# Drop the "Time" and "Class" columns
legit = legit.drop(['Time', 'Class'], axis=1)
fraud = fraud.drop(['Time', 'Class'], axis=1)

from sklearn.decomposition import PCA

pca = PCA(n_components=26, random_state=0)
legit_pca = pd.DataFrame(pca.fit_transform(legit), index=legit.index)
fraud_pca = pd.DataFrame(pca.transform(fraud), index=fraud.index)

legit_restored = pd.DataFrame(pca.inverse_transform(legit_pca), index=legit_pca.index)
fraud_restored = pd.DataFrame(pca.inverse_transform(fraud_pca), index=fraud_pca.index)

#function for measuring loss due to PCA.
import numpy as np

def get_anomaly_scores(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=df_original.index)
    return loss

legit_scores = get_anomaly_scores(legit, legit_restored)
fraud_scores = get_anomaly_scores(fraud, fraud_restored)

#plot the loss for the legitimate transactions
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

legit_scores.plot(figsize = (12, 6))
fraud_scores.plot(figsize = (12, 6))

#Plot the loss for the fraudulent transactions.
fraud_scores.plot(figsize = (12, 6))
legit_scores.plot(figsize = (12, 6))
