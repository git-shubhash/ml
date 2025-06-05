import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
data=fetch_california_housing()
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df.hist(figsize=(15,10),bins=30)
plt.tight_layout()
plt.show()
df.plot(kind='box',figsize=(15,10),layout=(3,3),subplots=True)
plt.tight_layout()
plt.show()
for col in df.columns:
    q1,q3=df[col].quantile([0.25,0.75])
    iqr=q3-q1
    out=df[(df[col]<q1-1.5*iqr)|(df[col]>q3+1.5*iqr)]
    print(col,"=",out.shape[0])
