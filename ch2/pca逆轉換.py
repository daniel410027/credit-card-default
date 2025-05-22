import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv('C:/Users/danie/Documents/spyder/creditcard.csv')
# data = df.drop(["Class", "Time", "Amount"],axis=1)
data = df

pca = PCA(n_components=31)
pca.fit(data)
data_original = np.dot(data, pca.components_) + pca.mean_

    
