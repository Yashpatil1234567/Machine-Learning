import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Load the Dataset
df = pd.read_csv('/content/drive/MyDrive/Machine Learning/tested.csv')
df.head()

# 2. Explore the data
df.describe()

# FIND MISSING DATA
print("\nMissing Values:")
df.isnull().sum()

# 3. Handle missing values of data
df.drop('Cabin', axis=1, inplace=True)

# Impute numerical column with mean and mode
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

imputer = SimpleImputer(strategy='most_frequent')
df['Fare'] = imputer.fit_transform(df[['Fare']])

df.dropna(subset=['Embarked'], inplace=True)
print("\nAfter Handling Missing Data:")
print(df.isnull().sum())

# 4. Handle Categorical Variables
categorical_columns = ['Sex', 'Embarked']
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# apply Label Encoder
class_mapping = {'S': 0, 'C': 1, 'Q': 2}
df['Embarked'] = df['Embarked'].map(class_mapping)

# apply one hot encoder
df = pd.get_dummies(df, columns=['Pclass'])
print("\nAfter Encoding Categorical Variables:")
print(df.head())

# 5. Apply Normalization and Standardization
numerical_columns = ['Age', 'Fare']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
print("\nAfter Normalization and Standardization:")
print(df.head())

# 6. Split the dataset
x = df.drop('Survived', axis=1)
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("\nTrain Set:")
print(x_train.head())
