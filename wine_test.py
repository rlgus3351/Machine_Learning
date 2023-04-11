import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

wine = load_wine()

wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

# print(wine_df.head())
X = wine_df
y = wine.target

# pd.plotting.scatter_matrix(wine_df, c=y)
# plt.figure(figsize=(20,10))
# plt.show()

train_acc = []
test_acc = []
n_range = range(1,134,1)

X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=20)

# test size = 30% [44 44 36]
# test size = 25% [46 49 38]
# print(np.bincount(y_train))

for i in n_range:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    train_acc.append(model.score(X_train,y_train))
    test_acc.append(model.score(X_test,y_test))

plt.figure(figsize=(20,10))
plt.plot(n_range, train_acc, label="train_acc" )
plt.plot(n_range, test_acc, label = 'test_acc')
plt.legend()
plt.grid()
plt.xticks(range(1,135,10))
plt.show()


