from google.colab import drive
drive.mount('/content/drive')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
import pandas as pd


# step 1 : load the dataset

df=pd.read_csv('/content/drive/MyDrive/Machine Learning/iris.csv')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
df.head()

# Filter only two classes for binary classification (Setosa vs Versicolor)
df = df[df['target'] != 'virginica']  # Removing Virginica class

# Convert target labels to numerical (0 and 1)
df['target'] = df['target'].map({'setosa': 0, 'versicolor': 1})

#STEP 2 : Data Preprocessing

from sklearn.preprocessing import StandardScaler
X = df.drop('target', axis=1)
y = df['target']
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#STEP 3 : SPLIT THE DATA

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#STEP 4 : Train SVM with Different Kernels

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

kernels = ['linear', 'poly', 'rbf']
accuracies = {}

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[kernel] = acc

print("Accuracy for each kernel:")
for k, v in accuracies.items():
    print(f"{k}: {v:.2f}")

# STEP 5 :Visualize Support Vectors and Margins (for 2-feature data)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use only 2 features for visualization
X_vis = X_scaled[:, :2]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.2, random_state=42)

# Train with linear kernel
svc = SVC(kernel='linear')
svc.fit(X_train_vis, y_train_vis)

# Plotting
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1')
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    plt.title(title)
    plt.legend()
    plt.show()

plot_decision_boundary(svc, X_train_vis, y_train_vis, 'SVM with Linear Kernel')


#STEP 6 : Perform Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)

# Evaluate best model
y_pred_best = grid.predict(x_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print("Best Accuracy after tuning:", best_accuracy)
