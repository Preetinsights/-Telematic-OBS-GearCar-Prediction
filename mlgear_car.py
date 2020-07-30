#!/usr/bin/env python
# coding: utf-8

# In[4]:


from IPython.display import Image
Image(filename='OBS-Flowchart.png')


# In[3]:


#Setting It Up
#I collected all of the data above and combined them into one dataframe.
#First, we will import the libraries we will be using and also load our data into a Pandas dataframe.

# Import needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sn
import sklearn

# Python magic to show plots inline in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import datetime as dt
from datetime import datetime
import math


# In[5]:


# Import data
df = pd.read_csv("https://raw.githubusercontent.com/Preetinsights/Telematic-OBS-Gear-Prediction/master/allcars.csv")


# In[7]:


#It is a good practice to understand the data first and try to gather as many insights from it. 
#EDA is all about making sense of data in hand,before getting them dirty with it.

#1. Check for Missing Data
#2. Summary statistics for measures of location and dispersion
#3. Heatmap for Correlations for Data Structure
#4. Establish a basis to uncover a parsimonious model, one which explains the data with a minimum number of predictor variables.

#...display the first 5 records in the data...#
df.head()


# In[8]:


#data atributes
df.info()


# In[9]:


#Check for Missing Data
df.isnull().values.any()


# In[11]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape


# In[12]:


#### Drop cells with NaN
df = df.dropna(axis=0,subset=['cTemp'])
df = df.dropna(axis=0,subset=['dtc'])
df = df.dropna(axis=0,subset=['iat'])
df = df.dropna(axis=0,subset=['imap'])
df = df.dropna(axis=0,subset=['tAdv'])


# In[13]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape


# In[14]:


#summary statistics
df.describe()


# In[15]:


# Seaborn doesn't handle NaN values, so we can fill them with 0 for now.
df = df.fillna(value=0)
# Pair grid of key variables.
g = sns.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Pairwise Grid of Numeric Features');


# In[16]:


g = sns.PairGrid(df, vars=["gps_speed", "speed"])
g = g.map(plt.scatter)


# In[17]:


g = sns.PairGrid(df, vars=["iat", "rpm"])
g = g.map(plt.scatter)
#iat is in-board automatic transmission
#rpm = revolution per minute


# In[18]:


#To use linear regression for modelling,its necessary to remove correlated variables to improve your model.
#One can find correlations using pandas “.corr()” function and can visualize the correlation matrix using a heatmap in seaborn.

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap='Blues')
plt.title('Correlation Heatmap of Numeric Features')


# In[19]:


#Select variables with complete dataset (no nan or zero)
df1 = pd.DataFrame(df,columns=['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed'])


# In[20]:


#Remove correlated variables before feature selection.
corrMatrix = df1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#Here, it can be infered that IAT – In-dash automatic transmission “iat” has strong positive correlation with circular temperature “cTemp”


# In[21]:


#Make final dataset
df.columns


# In[22]:


cols = df.columns.tolist()


# In[25]:


df1.to_csv('allcars.csv')


# In[26]:


df1.dtypes


# In[27]:


# FEATURE ENGINEERING
# Define custom function to create lag values
#Feature Engineering
#Currently our dataframe isn’t exactly what we need. This is because our machine learning algorithms only learn row by row and aren’t aware of other rows when learning or making predictions. We can overcome this challenge by imputing previous time value, or lags, into our data.
#After some trial and error, I determined that 4 lags (or 4 months) work best. The code below creates a function that will create 4 lags for each feature in the ‘features’ list. Our new dataframe now has 41 columns!
#...Lag Transformation is an FE technique...#, seconded by Feature Split
def feature_lag(features):
    for feature in features:
        df[feature + '-lag1'] = df[feature].shift(1)
        df[feature + '-lag2'] = df[feature].shift(2)
        df[feature + '-lag3'] = df[feature].shift(3)
        df[feature + '-lag4'] = df[feature].shift(4)

# Define columns to create lags for
features = ['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed']

# Call custom function
feature_lag(features)

#Feature engineering is the process of transforming raw data into features 
#that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data. 
#Feature engineering turn your inputs into things the algorithm can understand


# In[28]:


#predict gps speed 3, 6, and 12 months ahead.
df1['y3'] = df.gps_speed.shift(-3)
df1['y6'] = df.gps_speed.shift(-6)
df1['y12'] = df.gps_speed.shift(-12)


# In[29]:


df1 = df1.dropna(axis=0,subset=['y3'])
df1 = df1.dropna(axis=0,subset=['y6'])
df1 = df1.dropna(axis=0,subset=['y12'])


# In[30]:


df1.dtypes


# In[31]:


#Split into Training and Test Data
#Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

y = df1['gps_speed']


# In[32]:


X = df1.drop(['gps_speed'], axis=1)


# In[33]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)


# In[34]:


X_train.shape, y_train.shape


# In[35]:


X_test.shape, y_test.shape


# In[36]:


X.columns


# In[37]:


df1.dtypes


# In[38]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
# Create linear regression object
regr = LinearRegression()


# In[39]:


# Train the model using the training sets
regr.fit(X_train, y_train)


# In[40]:


# Make predictions using the testing set
lin_pred = regr.predict(X_test)


# In[41]:


linear_regression_score = regr.score(X_test, y_test)
linear_regression_score


# In[42]:


linear_regression_score = regr.score(X_train, y_train)
linear_regression_score


# In[43]:


from math import sqrt
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lin_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lin_pred))


# In[44]:


plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()


# In[45]:


### Neural Network Regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create MLPRegressor object
mlp = MLPRegressor()


# In[46]:


# Train the model using the training sets
mlp.fit(X_train, y_train)


# In[47]:


# Score the model
neural_network_regression_score = mlp.score(X_test, y_test)
neural_network_regression_score


# In[48]:


# Score the model
neural_network_regression_score = mlp.score(X_train, y_train)
neural_network_regression_score


# In[49]:


# Make predictions using the testing set
nnr_pred = mlp.predict(X_test)


# In[50]:


# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, nnr_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, nnr_pred))


# In[51]:


plt.scatter(y_test, nnr_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Neural Network Regression Predicted vs Actual')
plt.show()


# In[52]:


###Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()


# In[53]:


lasso.fit(X_train, y_train)


# In[54]:


# Score the model
lasso_score = lasso.score(X_test, y_test)
lasso_score


# In[55]:


# Score the model
lasso_score = lasso.score(X_train, y_train)
lasso_score


# In[56]:


# Make predictions using the testing set
lasso_pred = lasso.predict(X_test)


# In[57]:


print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))

# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lasso_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lasso_pred))


# In[58]:


plt.scatter(y_test, lasso_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Lasso Predicted vs Actual')
plt.show()


# In[59]:


##ElasticNet
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)


# In[60]:


elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score


# In[61]:


elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score


# In[62]:


elasticnet_pred = elasticnet.predict(X_test)


# In[63]:


# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, elasticnet_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, elasticnet_pred))


# In[64]:


###Decision Forest Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=200, random_state=1234)


# In[65]:


# Train the model using the training sets
regr_rf.fit(X_train, y_train)


# In[66]:


regr_rf.fit(X_test, y_test)


# In[67]:


# Score the model
decision_forest_score = regr_rf.score(X_test, y_test)
decision_forest_score


# In[68]:


# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)


# In[69]:


from math import sqrt
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_rf_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, regr_rf_pred))


# In[70]:


features = X.columns
importances = regr_rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

#Random Forest Regression show how useful they are at predicting the target variable or improve the accuracy of the model


# In[71]:


plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()


# In[72]:


#Extra Trees Regression

from sklearn.ensemble import ExtraTreesRegressor
extra_tree = ExtraTreesRegressor(n_estimators=200, random_state=1234)


# In[73]:


extra_tree.fit(X_train, y_train)


# In[74]:


extratree_score = extra_tree.score(X_test, y_test)
extratree_score


# In[75]:


extratree_score = extra_tree.score(X_train, y_train)
extratree_score


# In[76]:


extratree_pred = extra_tree.predict(X_test)


# In[77]:


print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, extratree_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, extratree_pred))


# In[78]:


features = X.columns
importances = extra_tree.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

#Extra Trees Regression show how useful they are at predicting the target variable or improve the accuracy of the model


# In[79]:


plt.scatter(y_test, extratree_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Extra Trees Predicted vs Actual')
plt.show()


# In[80]:


#Evaluate Models
print("Scores:")
print("Linear regression score: ", linear_regression_score)
print("Neural network regression score: ", neural_network_regression_score)
print("Lasso regression score: ", lasso_score)
print("ElasticNet regression score: ", elasticnet_score)
print("Decision forest score: ", decision_forest_score)
print("Extra Trees score: ", extratree_score)
print("\n")
print("RMSE:")
print("Linear regression RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
print("Neural network RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
print("Lasso RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))
print("ElasticNet RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
print("Decision forest RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
print("Extra Trees RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))


# In[82]:


from IPython.display import Image
Image(filename='mmv.png')


# In[83]:


import pandas as pd
"""
A framework script that tags the data points according to the gear and assigns it a color and plots the data. 
The gear detection is done by assuming the borders generated using any of the algorithms and placed in
the borders array. 
"""

get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt


def get_gear(entry, borders):
    if entry['rpm'] == 0:
        return 0
    rat = entry['speed'] / entry['rpm'] * 1000
    if np.isnan(rat) or np.isinf(rat):
        return 0
    for i in range(0, len(borders)):
        if rat < borders[i] :
            return i + 1
    return 0

num_trips = 10
# Import data
df1 = pd.DataFrame(df,columns=['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed'])
obddata = df[df['tripID']<num_trips]

# borders = get_segment_borders(obddata)
borders = [7.070124715964856, 13.362448319790191, 19.945056624926686, 27.367647318253834, 32.17327586520911]
# The segment borders represents spaces in the data.  
#the borders creates overlaps that will show the direction of gear adjustments as the machine moves and speed.
obddata_wgears = obddata
obddata_wgears['gear'] = obddata.apply(lambda x : get_gear(x, borders), axis=1)

# print(obddata_wgears)

colors = [x * 50 for x in obddata_wgears['gear']]
plt.scatter(obddata_wgears['rpm'], obddata_wgears['speed'], c=colors)
plt.plot()

