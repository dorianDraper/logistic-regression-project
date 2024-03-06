import pandas as pd
import numpy as np

path = pd.read_csv('C:/Users/Jorge Payà/Desktop/4Geeks/DSML Bootcamp/Projects/logistic-regression-project/data/raw/data.csv', sep=';')
total_data = pd.DataFrame(path)
print(total_data.head())

# check if there are duplicated rows
total_data.duplicated()
print('There are a total of', total_data.duplicated().sum(),'duplicated rows')
total_data[total_data.duplicated(keep=False)]
# remove the duplicated rows
total_data = total_data.drop_duplicates()
print('After we remove the duplicated ones the shape of the data is', total_data.shape)

# check if there are missing values
print('There are a total of', total_data.isnull().sum().sum(),'missing values')
print(total_data.isnull().sum())
print('*'*50)
print('*'*50)

# Min-Max scaling
from sklearn.preprocessing import MinMaxScaler

total_data["job_n"] = pd.factorize(total_data["job"])[0]
total_data["marital_n"] = pd.factorize(total_data["marital"])[0]
total_data["education_n"] = pd.factorize(total_data["education"])[0]
total_data["default_n"] = pd.factorize(total_data["default"])[0]
total_data["housing_n"] = pd.factorize(total_data["housing"])[0]
total_data["loan_n"] = pd.factorize(total_data["loan"])[0]
total_data["contact_n"] = pd.factorize(total_data["contact"])[0]
total_data["month_n"] = pd.factorize(total_data["month"])[0]
total_data["day_of_week_n"] = pd.factorize(total_data["day_of_week"])[0]
total_data["poutcome_n"] = pd.factorize(total_data["poutcome"])[0]
total_data["y_n"] = pd.factorize(total_data["y"])[0]
num_variables = ["job_n", "marital_n", "education_n", "default_n", "housing_n", "loan_n", "contact_n", "month_n", "day_of_week_n", "poutcome_n", "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y_n"]

scaler = MinMaxScaler()
scal_features = scaler.fit_transform(total_data[num_variables])
total_data_scal = pd.DataFrame(scal_features, index = total_data.index, columns= num_variables)
print(total_data_scal.head())

# Feature selection
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = total_data_scal.drop("y_n", axis=1) # drop the target variable from the data 
y = total_data_scal["y_n"] # select the target variable 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) # split the data into training and testing sets 

selection_model = SelectKBest(chi2, k=5) # select the 5 best features according to chi2 test 
selection_model.fit(X_train, y_train) # fit the model to the data 
ix = selection_model.get_support() # get the indices of the selected features 
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix]) # transform the data to the selected features 
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix]) # transform the data to the selected features
print(X_train_sel.head()) # print the first 5 rows of the selected features
print('*'*50)
print(X_test_sel.head()) # print the first 5 rows of the selected features

X_train_sel["y_n"] = list(y_train) # add the target variable to the selected features 
X_test_sel["y_n"] = list(y_test) # add the target variable to the selected features
X_train_sel.to_csv('C:/Users/Jorge Payà/Desktop/4Geeks/DSML Bootcamp/Projects/logistic-regression-project/data/processed/clean_train.csv', index=False) # save the training data to a csv file
X_test_sel.to_csv('C:/Users/Jorge Payà/Desktop/4Geeks/DSML Bootcamp/Projects/logistic-regression-project/data/processed/clean_test.csv', index=False) # save the testing data to a csv file




