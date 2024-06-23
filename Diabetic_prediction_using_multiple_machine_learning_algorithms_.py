# importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data= pd.read_csv(r"D:\WORK\Diabetic-prediction-using-multiple-machine-learning-algorithms\diabetes_prediction_dataset.csv")

data.head()

# checking null values
data.isnull()

# describe data set
data.describe()
#
data.shape

plt.figure(figsize=(8,6))
plt.hist(data['age'], bins=20, edgecolor="black", color="cyan")
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Defining colors for every gender catagory
colors = {'Male': 'blue', 'Female': 'red', 'Other': 'green'}

plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=data, palette=colors)
plt.title('Count Plot of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Distribution of HbA1c Level
plt.figure(figsize=(8, 6))
sns.histplot(data['HbA1c_level'],kde=True, color= 'green')
plt.title('Distribution of HbA1c Level')
plt.xlabel('HbA1c_level')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Dertining BMI at Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y= 'bmi', data=data)
plt.title('Boxplot of BMI at Gender')
plt.xlabel('Gender')
plt.ylabel('BMI')
plt.grid(True)
plt.show

# Defineing the color palatte for several smoking history categories
avg_glucose_by_smoking = data.groupby('smoking_history')['blood_glucose_level'].mean().reset_index()
sns.barplot(x='smoking_history', y='blood_glucose_level', data=avg_glucose_by_smoking)
plt.title('Average Blood Glucose Level by Smoking History')
plt.xlabel('smoking History')
plt.ylabel('Average Blood Glucose Level')
plt.grid(True)
plt.show()

data.head()

from sklearn.preprocessing import LabelEncoder

#transforming categorical to numerical columns
data ['gender'] = LabelEncoder().fit_transform(data ['gender'])
data ['smoking_history'] = LabelEncoder().fit_transform(data ['smoking_history'])
data.head()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# separating the features (x) and target variable(y)
x = data.drop('blood_glucose_level', axis=1) # features
y = data['blood_glucose_level'] # target variable

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# standardizing the featurs
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Training the linear regresiion model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# making predictions on the testing set
y_pred = model.predict(x_test_scaled)

# evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')



# Support Vector Regression (SVR)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# seperating features (x) and target variable (y)
x = data.drop('blood_glucose_level') #features
y = data['blood_glucose_level'] #target variable

#splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# standardizing the features (recommended for SVR)
scalar = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Training the SVR model
model = SVR(kernel='rbf') # radial basis function (RBF) kernel is commonly used
model.fit(x_train_scaled, y_train)

# making predictions on the testing set
y_pred = model.predict(x_test_scaled)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Absolute ERROR (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R^2 Score:{r2:.4f}')