import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df=pd.read_csv("iris.csv")
x=df.iloc[:,:-1].values
y_lablel=df.iloc[:,-1].astype("category")
y=y_lablel.cat.codes
xpca=PCA(n_components=2).fit_transform(x)

for label in np.unique(y):
    plt.scatter(xpca[y==label,0])