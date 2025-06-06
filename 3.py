import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("iris.csv") 
X = df.iloc[:, :-1].values  
y_labels = df.iloc[:, -1].astype('category')
y = y_labels.cat.codes  
X_pca = PCA(n_components=2).fit_transform(X)
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=y_labels.cat.categories[label])
plt.legend()
plt.show()
