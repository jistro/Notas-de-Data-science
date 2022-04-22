import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv('Data.csv')
print(ds)
X=ds.iloc[:,:-1].values #variable independiente
y=ds.iloc[:,3].values   #variable dependeiente

#limpiar datos nan (tratamiento de los NaN's)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

