#importing labraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Data
df = pd.read_csv("train.csv")
df

#EDA
df.isnull().sum()
df.info()
df.duplicated().sum()
df.describe()

df['price_range'].value_counts()

#Visualization
sns.countplot(x = df["price_range"])

plt.figure(figsize=(15,14))
sns.heatmap(df.corr() , annot=True , cmap="inferno");
df.hist(figsize=(15,15))
plt.show()

sns.barplot(x = df["price_range"], y =df['battery_power'], data= df)

plt.figure(figsize= (14,6))
plt.subplot(1, 2, 1)
sns.barplot(x = df["price_range"], y =df['px_height'], data= df, palette='pink')
plt.subplot(1, 2, 2)
sns.barplot(x = df["price_range"], y =df['px_width'], data= df, palette='ocean')
plt.show()

sns.barplot(x = df["price_range"], y =df['ram'], data= df, palette='ocean')

#Training the DataFrame
from sklearn.model_selection import train_test_split
X = df.drop("price_range" , axis=1)
y = df.price_range


X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.25 , random_state=18 , stratify=y)

X_train.shape , y_train.shape

#Data Preprocessing
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#Models for training
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

models = [
    ('LogisticRegression', LogisticRegression()),
    ('SVM', SVC()), 
    ('RandomForest', RandomForestClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('DecisionTree', DecisionTreeClassifier()),
    ('NaiveBayes', GaussianNB())
]

for name, model in models:
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate accuracy 
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print accuracy
    print(f'{name} test accuracy: {accuracy:.3f}')
    
#Data Preprocessing     
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#Fit the model
from sklearn.linear_model import LogisticRegression

# Create a LogisticRegression model
logistic_reg = LogisticRegression()

# Fit the model to the training data
logistic_reg.fit(X_train, y_train)

# Make predictions on the test data
logistic_reg_predictions =logistic_reg.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, logistic_reg_predictions)
print("Accuracy of Logistic Regression:", accuracy)

#Predicting Model
prediction = logistic_reg.predict(X_test)

#Predicting Model on Test Data
test_df = pd.read_csv(r"c:\Users\Hp\Downloads\archive\test.csv") 
test_df

test_df = test_df.drop(['id'], axis = 1)
test_df.shape

X_test_scaled = scalar.transform(test_df)

testPrediction= logistic_reg.predict(X_test_scaled)

test_df['predicted_price'] = testPrediction
test_df
