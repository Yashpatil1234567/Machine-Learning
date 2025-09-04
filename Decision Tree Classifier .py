# STEP 0: Import Required Libraries
from google.colab import drive
drive.mount('/content/drive')  # Mount Google Drive to access files

import pandas as pd           
import numpy as np            
import matplotlib.pyplot as plt  
import seaborn as sns         
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from sklearn import tree       
# ---------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ---------------------------------------------------------
df = pd.read_csv('/content/drive/MyDrive/Machine Learning/seattle-weather.csv')
print("First 5 rows of dataset:")
print(df.head())

# ---------------------------------------------------------
# STEP 2: DATA PREPROCESSING
# ---------------------------------------------------------
print("Missing values per column:")
print(df.isnull().sum())  # Check if any column has missing values

# Encode categorical column 'weather' into numeric codes
# Example: drizzle=0, fog=1, rain=2, snow=3, sun=4
for col in ['weather']:
    df[col] = df[col].astype('category').cat.codes
    df[col] = df[col].astype('int64')

print("\nData after encoding categorical features:")
print(df.head())

# ---------------------------------------------------------
# STEP 3: SPLIT THE DATA
# ---------------------------------------------------------
# Convert 'date' column to datetime format (so we can extract year, month, day)
df['date'] = pd.to_datetime(df['date'])

# Extract useful features from date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Features (X) → all columns except 'weather' and 'date'
X = df.drop(['weather', 'date'], axis=1)
y = df['weather']

# Split dataset into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# STEP 4: TRAIN DECISION TREE (WITHOUT PRUNING)
# ---------------------------------------------------------
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)  
# Predictions on test data
y_pred = clf.predict(X_test)

# ---------------------------------------------------------
# VISUALIZE THE TREE STRUCTURE (WITHOUT PRUNING)
# ---------------------------------------------------------
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(clf,
          feature_names=X.columns,  
          class_names=['drizzle', 'fog', 'rain', 'snow', 'sun'], 
          filled=True,             
          rounded=True,            
          fontsize=1)                
plt.title("Decision Tree Visualization (Without Pruning)")
plt.show()

# ---------------------------------------------------------
# STEP 6: TRAIN DECISION TREE WITH PRUNING
# ---------------------------------------------------------

clf_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)

# Train pruned tree
clf_pruned.fit(X_train, y_train)

# Visualize pruned tree
plt.figure(figsize=(10,8))
plot_tree(clf_pruned,
          feature_names=X.columns,
          class_names=['drizzle', 'fog', 'rain', 'snow', 'sun'],
          filled=True,
          rounded=True,
          fontsize=3)
plt.title("Pruned Decision Tree Visualization")
plt.show()

# ---------------------------------------------------------
# STEP 7: COMPARE PERFORMANCE
# ---------------------------------------------------------
# Predictions for both models
y_pred_no_prune = y_pred
y_pred_pruned = clf_pruned.predict(X_test)

# Accuracy scores
acc_no_prune = accuracy_score(y_test, y_pred_no_prune)
acc_pruned = accuracy_score(y_test, y_pred_pruned)

print(f"Accuracy without pruning: {acc_no_prune:.2f}")
print(f"Accuracy with pruning: {acc_pruned:.2f}")

# ---------------------------------------------------------
# STEP 8: ANALYZE FEATURE IMPORTANCE
# ---------------------------------------------------------
# Feature importance tells us which features are most important


# For unpruned tree
importances_no_prune = pd.Series(clf.feature_importances_, index=X.columns)

# For pruned tree
importances_pruned = pd.Series(clf_pruned.feature_importances_, index=X.columns)

# Plot feature importance (Unpruned)
plt.figure(figsize=(10, 6))
importances_no_prune.sort_values().plot(kind='barh', title='Feature Importances (No Pruning)')
plt.show()

# Plot feature importance (Pruned)
plt.figure(figsize=(10, 6))
importances_pruned.sort_values().plot(kind='barh', title='Feature Importances (With Pruning)')
plt.show()

# Print numerical values of importance
print("Feature importances (No Pruning):")
print(importances_no_prune.sort_values(ascending=False))

print("\nFeature importances (With Pruning):")
print(importances_pruned.sort_values(ascending=False))
