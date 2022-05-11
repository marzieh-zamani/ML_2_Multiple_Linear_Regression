# Project 2: Multiple Linear Regression
Multiple linear regression on Fish.csv dataset.

## Code Breakdown
### Step 1: Define function_1 to estimate the coefficients (weights) for multiple regression, return the training & testing RSS
`function_1` implements the gradient descent algorithm (regression_gradient_descent) from scratch to compute coefficients for multiple regression. Also, returns the training RSS, and testing RSS.
`weight, train_rss_history, test_rss_history =`
    `function_1(train_feature, train_output, test_feature, test_output, weight0, learning_rate, tolerance)`

### Step 2: Define function_2 to predict the model output
`function_2` predicts the output using the calculated weights.
`predicted_output = function_2(input_feature, weight)`

### Step 3: Define function_3 to calculate the model RSS
`function_3` calculates the RSS.
`RSS = function_3(input_feature, output, weight)`

### Step 4: Import fish data from Fish.csv
1. Importing fish data from Fish.csv using panda.read_csv function;
2. Converting Panda data to Numpy data;
3. Random spliting the data into 80% training and 20% test data (train_feature, test_feature, train_output, test_output);
4. Saving input features and output weigth separately;

### Step 5: Use function_1 to estimate coefficients, train & test RSS for models
Use the above functions to compute the model coefficients, train and test error (RSS) for each of the following cases. 
Also, calculate training and test RSS for each step of the gradient descent and then plot it for each of the cases given below.

**Model (1)**
model features: ‘Length1’
output: ‘Weight’
initial weights: [-7.5, 1] (intercept, Length1 respectively)
step size (learning rate) = 7e-10
tolerance = 1.4e4

**Model (2)**
model features = ‘Length1’, ‘Width’
output = ‘Weight’
initial weights = [-8.5, 1, 1] (intercept, Length1 and Width respectively)
step size (learning rate) = 4e-10
tolerance = 1.4e4

**Model (3)**
model features = ‘Length1’, 'Width', 'Height'
output = ‘Weight’
initial weights = [-10, 1, 1,1] (intercept, Length1, Width, Height respectively)
step size (learning rate) = 4e-10
tolerance = 1.4e4

### Step 6: Implement Scikit Learn linear regression functions to obtain predicted output, train & test RSS
Use in-built linear regression functions of Scikit Learn library to compute higher polynomial regression models for degrees 2, 3, 4, 5 and 6. 
Use ‘Lenght1’ as the input feature and ‘Weight’ as output. 
For each of the model, compute the RSS (on the train and test dataset), and plot the model through the training data.

### Step 7: Compare and plot Train & Test RSS vs. Model degree
Compare and plot the Train & Test RSS vs. model degree for all models.

## Main files to check
The main file to check is the Jupyter notebook where:
- The functions are defined;
- The data is given;
- Then, the functions are called;
- The results are displayed and saved.

## Setup
Install [Miniconda](https://conda.io/miniconda).
Then, run the jupyter notebook in the "code" folder.

## Acknowledgment and References
This project has been developed based on the assignment provided by Dr. Abdul Bais, P.Eng. (abdul.bais@uregina.ca), my instructor for the course “ENEL-865/ENSE 865: Applied Machine Learning”.

This assignment is based on the first assignment of Machine Learning Regression course (from Coursera). 

- Dr. Abdul Bais, P.Eng. (abdul.bais@uregina.ca) page: 
https://www.uregina.ca/engineering/faculty-staff/faculty/bais-abdul.html

## Dataset
Download the Fish Market Dataset from Kaggle (https://www.kaggle.com/aungpyaeap/fish-market) to estimate weight of fish. Random split data into 80% training and 20% test data.

## My contribution
All scripts are written by my self.
______________
Marzieh Zamani