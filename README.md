# final_project
TOPIC - Diabetic prediction using multiple machine learning algorithms

Dataset - https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?resource=download 

## Diabetes Prediction Analysis with Machine Learning

This repository contains Python code for analyzing a diabetes prediction dataset and building machine learning models to predict diabetes diagnosis.

### Dependencies

This code requires the following Python libraries:

* pandas
* numpy
* matplotlib
* seaborn
* sklearn (including sub-libraries for preprocessing, model selection, and evaluation)

### Usage

1. **Clone the repository or download the code.**
2. **Ensure you have the required libraries installed.** Use `pip install <library_name>` to install them.
3. **Run the script.** Execute the Python script (`diabetes_prediction_analysis.py` or similar) in your terminal or a suitable development environment.

### Code Structure

The script performs the following steps:

1. **Data Loading:** It imports the diabetes prediction dataset using pandas.
2. **Exploratory Data Analysis (EDA):**
    * It checks for missing values and displays basic statistics of the data.
    * It visualizes the distribution of features using histograms, boxplots, and countplots.
3. **Data Preprocessing:**
    * It encodes categorical features (e.g., gender) into numerical values using LabelEncoder.
    * It splits the data into features (X) and target variable (y).
    * It further splits the data into training and testing sets for model training and evaluation.
    * It performs feature scaling (standardization) as recommended for some models.
4. **Model Building and Evaluation:**
    * **Linear Regression:** It trains a linear regression model and evaluates its performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R^2 score.
    * **Support Vector Regression (SVR):** It trains an SVR model with a radial basis function (RBF) kernel and evaluates its performance using the same metrics.
    * **Logistic Regression:** It trains a logistic regression model and evaluates its performance using accuracy, confusion matrix, and classification report.
    * **Random Forest Classifier:** It trains a random forest classifier and evaluates its performance using the same metrics as logistic regression.
5. **Model Comparison:** It compares the accuracy of the different models using a bar chart.

### Results

The script outputs the performance metrics (MAE, MSE, R^2, accuracy) for each model and displays confusion matrices and a classification report for the classification models. 