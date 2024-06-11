import pandas as pd
import numpy as np
import plotly.express as px
from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]
iris = pd.read_csv(filename)

print(iris.head())
print(iris.describe())
print("Target Labels",iris["species"].unique())

fig = px.scatter(iris,x="sepal_width",y="sepal_length",color="species")
fig.show()

x = iris.drop("species",axis=1)
y = iris["species"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

x_new = pd.DataFrame([[5,2.9,1,0.2]],columns=x.columns)
prediction = knn.predict(x_new)
print()
print("PREDICTION : {}".format(prediction))
