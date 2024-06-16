# importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv(r"D:\WORK\Diabetic-prediction-using-multiple-machine-learning-algorithms\diabetes_prediction_dataset.csv")

data.head()

# checking null values
data.isnull()

# describe data set
data.describe()

data.shape

plt.figure(figsize=(8,6))
plt.hist(data['age'], bins=20, edgecolor="black", color="cyan")
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()