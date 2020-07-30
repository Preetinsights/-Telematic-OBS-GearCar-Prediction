

```python
from IPython.display import Image
Image(filename='OBS-Flowchart.png')
```




![png](output_0_0.png)




```python
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
%matplotlib inline
plt.style.use('ggplot')
import datetime as dt
from datetime import datetime
import math
```


```python
# Import data
df = pd.read_csv("https://raw.githubusercontent.com/Preetinsights/Telematic-OBS-Gear-Prediction/master/allcars.csv")
```


```python
#It is a good practice to understand the data first and try to gather as many insights from it. 
#EDA is all about making sense of data in hand,before getting them dirty with it.

#1. Check for Missing Data
#2. Summary statistics for measures of location and dispersion
#3. Heatmap for Correlations for Data Structure
#4. Establish a basis to uncover a parsimonious model, one which explains the data with a minimum number of predictor variables.

#...display the first 5 records in the data...#
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>timeStamp</th>
      <th>tripID</th>
      <th>accData</th>
      <th>gps_speed</th>
      <th>battery</th>
      <th>cTemp</th>
      <th>dtc</th>
      <th>eLoad</th>
      <th>iat</th>
      <th>imap</th>
      <th>kpl</th>
      <th>maf</th>
      <th>rpm</th>
      <th>speed</th>
      <th>tAdv</th>
      <th>tPos</th>
      <th>deviceID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>117459</td>
      <td>184</td>
      <td>28:21.0</td>
      <td>0</td>
      <td>0f18fe2806d00210bf030fc1fe0ebffe0ec0fd10c0ff0e...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>117460</td>
      <td>185</td>
      <td>28:22.0</td>
      <td>0</td>
      <td>0f48fe400660fe0dc1ff0ebfff0fc0010ebefd0dc0010f...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>117461</td>
      <td>186</td>
      <td>28:23.0</td>
      <td>0</td>
      <td>0ef8fe300678fe0ebfff0ec0030fc0ff0dc1000fc0000e...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>117462</td>
      <td>187</td>
      <td>28:24.0</td>
      <td>0</td>
      <td>0f20fe2806d8ff0cc0ff0dc2000fc1ff0ec1010dbe000e...</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>117463</td>
      <td>188</td>
      <td>28:25.0</td>
      <td>0</td>
      <td>0f50fe800678fe10c0000ec0000ec0000ebf0110c00010...</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#data atributes
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 106460 entries, 0 to 106459
    Data columns (total 19 columns):
    Unnamed: 0      106460 non-null int64
    Unnamed: 0.1    106460 non-null int64
    timeStamp       106460 non-null object
    tripID          106460 non-null int64
    accData         106460 non-null object
    gps_speed       106460 non-null float64
    battery         106460 non-null float64
    cTemp           106460 non-null float64
    dtc             106460 non-null float64
    eLoad           106460 non-null float64
    iat             106460 non-null float64
    imap            106460 non-null float64
    kpl             106460 non-null float64
    maf             106460 non-null float64
    rpm             106460 non-null float64
    speed           106460 non-null float64
    tAdv            106460 non-null float64
    tPos            106460 non-null float64
    deviceID        106460 non-null int64
    dtypes: float64(13), int64(4), object(2)
    memory usage: 15.4+ MB
    


```python
#Check for Missing Data
df.isnull().values.any()
```




    False




```python
# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape
```




    (0, 19)




```python
#### Drop cells with NaN
df = df.dropna(axis=0,subset=['cTemp'])
df = df.dropna(axis=0,subset=['dtc'])
df = df.dropna(axis=0,subset=['iat'])
df = df.dropna(axis=0,subset=['imap'])
df = df.dropna(axis=0,subset=['tAdv'])
```


```python
# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape
```




    (0, 19)




```python
#summary statistics
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>tripID</th>
      <th>gps_speed</th>
      <th>battery</th>
      <th>cTemp</th>
      <th>dtc</th>
      <th>eLoad</th>
      <th>iat</th>
      <th>imap</th>
      <th>kpl</th>
      <th>maf</th>
      <th>rpm</th>
      <th>speed</th>
      <th>tAdv</th>
      <th>tPos</th>
      <th>deviceID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.0</td>
      <td>106460.000000</td>
      <td>106460.0</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.0</td>
      <td>106460.0</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.0</td>
      <td>106460.0</td>
      <td>106460.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>170703.476893</td>
      <td>53428.476893</td>
      <td>63.697849</td>
      <td>18.222948</td>
      <td>0.0</td>
      <td>64.143575</td>
      <td>0.0</td>
      <td>35.477576</td>
      <td>31.122901</td>
      <td>96.442175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1183.945900</td>
      <td>33.075089</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30732.539133</td>
      <td>30732.539133</td>
      <td>38.719864</td>
      <td>18.727147</td>
      <td>0.0</td>
      <td>29.107386</td>
      <td>0.0</td>
      <td>22.502089</td>
      <td>15.797552</td>
      <td>47.344598</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>759.576518</td>
      <td>33.972104</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>117459.000000</td>
      <td>184.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>144088.750000</td>
      <td>26813.750000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>60.000000</td>
      <td>0.0</td>
      <td>22.352941</td>
      <td>24.000000</td>
      <td>97.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>800.750000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>170703.500000</td>
      <td>53428.500000</td>
      <td>60.000000</td>
      <td>13.600000</td>
      <td>0.0</td>
      <td>80.000000</td>
      <td>0.0</td>
      <td>38.823529</td>
      <td>34.000000</td>
      <td>99.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1167.750000</td>
      <td>25.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>197318.250000</td>
      <td>80043.250000</td>
      <td>99.000000</td>
      <td>27.800000</td>
      <td>0.0</td>
      <td>81.000000</td>
      <td>0.0</td>
      <td>48.235294</td>
      <td>43.000000</td>
      <td>110.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1733.500000</td>
      <td>50.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>223933.000000</td>
      <td>106658.000000</td>
      <td>126.000000</td>
      <td>82.100000</td>
      <td>0.0</td>
      <td>84.000000</td>
      <td>0.0</td>
      <td>94.901961</td>
      <td>58.000000</td>
      <td>221.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3566.000000</td>
      <td>149.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Seaborn doesn't handle NaN values, so we can fill them with 0 for now.
df = df.fillna(value=0)
# Pair grid of key variables.
g = sns.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Pairwise Grid of Numeric Features');
```


![png](output_10_0.png)



```python
g = sns.PairGrid(df, vars=["gps_speed", "speed"])
g = g.map(plt.scatter)
```


![png](output_11_0.png)



```python
g = sns.PairGrid(df, vars=["iat", "rpm"])
g = g.map(plt.scatter)
#iat is in-board automatic transmission
#rpm = revolution per minute
```


![png](output_12_0.png)



```python
#To use linear regression for modelling,its necessary to remove correlated variables to improve your model.
#One can find correlations using pandas “.corr()” function and can visualize the correlation matrix using a heatmap in seaborn.

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap='Blues')
plt.title('Correlation Heatmap of Numeric Features')
```




    Text(0.5,1,'Correlation Heatmap of Numeric Features')




![png](output_13_1.png)



```python
#Select variables with complete dataset (no nan or zero)
df1 = pd.DataFrame(df,columns=['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed'])
```


```python
#Remove correlated variables before feature selection.
corrMatrix = df1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#Here, it can be infered that IAT – In-dash automatic transmission “iat” has strong positive correlation with circular temperature “cTemp”
```


![png](output_15_0.png)



```python
#Make final dataset
df.columns
```




    Index(['Unnamed: 0', 'Unnamed: 0.1', 'timeStamp', 'tripID', 'accData',
           'gps_speed', 'battery', 'cTemp', 'dtc', 'eLoad', 'iat', 'imap', 'kpl',
           'maf', 'rpm', 'speed', 'tAdv', 'tPos', 'deviceID'],
          dtype='object')




```python
cols = df.columns.tolist()
```


```python
df1.to_csv('allcars.csv')
```


```python
df1.dtypes
```




    tripID         int64
    gps_speed    float64
    cTemp        float64
    eLoad        float64
    iat          float64
    imap         float64
    rpm          float64
    speed        float64
    dtype: object




```python
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
```


```python
#predict gps speed 3, 6, and 12 months ahead.
df1['y3'] = df.gps_speed.shift(-3)
df1['y6'] = df.gps_speed.shift(-6)
df1['y12'] = df.gps_speed.shift(-12)
```


```python
df1 = df1.dropna(axis=0,subset=['y3'])
df1 = df1.dropna(axis=0,subset=['y6'])
df1 = df1.dropna(axis=0,subset=['y12'])
```


```python
df1.dtypes
```




    tripID         int64
    gps_speed    float64
    cTemp        float64
    eLoad        float64
    iat          float64
    imap         float64
    rpm          float64
    speed        float64
    y3           float64
    y6           float64
    y12          float64
    dtype: object




```python
#Split into Training and Test Data
#Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

y = df1['gps_speed']
```


```python
X = df1.drop(['gps_speed'], axis=1)
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)
```


```python
X_train.shape, y_train.shape
```




    ((74513, 10), (74513,))




```python
X_test.shape, y_test.shape
```




    ((31935, 10), (31935,))




```python
X.columns
```




    Index(['tripID', 'cTemp', 'eLoad', 'iat', 'imap', 'rpm', 'speed', 'y3', 'y6',
           'y12'],
          dtype='object')




```python
df1.dtypes
```




    tripID         int64
    gps_speed    float64
    cTemp        float64
    eLoad        float64
    iat          float64
    imap         float64
    rpm          float64
    speed        float64
    y3           float64
    y6           float64
    y12          float64
    dtype: object




```python
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
# Create linear regression object
regr = LinearRegression()
```


```python
# Train the model using the training sets
regr.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
# Make predictions using the testing set
lin_pred = regr.predict(X_test)
```


```python
linear_regression_score = regr.score(X_test, y_test)
linear_regression_score
```




    0.9925489189517546




```python
linear_regression_score = regr.score(X_train, y_train)
linear_regression_score
```




    0.9923194367832849




```python
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
```

    Coefficients: 
     [-4.80717894e-04  4.17246956e-02 -1.85684512e-02 -9.45400424e-03
     -1.56901951e-02 -1.03784532e-03  5.86184055e-01  5.43109400e-02
     -1.18445019e-01  3.86888018e-02]
    Root mean squared error: 1.62
    Mean absolute error: 0.99
    R-squared: 0.99
    


```python
plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()
```


![png](output_37_0.png)



```python
### Neural Network Regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create MLPRegressor object
mlp = MLPRegressor()
```


```python
# Train the model using the training sets
mlp.fit(X_train, y_train)
```




    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
# Score the model
neural_network_regression_score = mlp.score(X_test, y_test)
neural_network_regression_score
```




    0.9919201262827668




```python
# Score the model
neural_network_regression_score = mlp.score(X_train, y_train)
neural_network_regression_score
```




    0.9917593481132061




```python
# Make predictions using the testing set
nnr_pred = mlp.predict(X_test)
```


```python
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, nnr_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, nnr_pred))
```

    Root mean squared error: 1.69
    Mean absolute error: 1.03
    R-squared: 0.99
    


```python
plt.scatter(y_test, nnr_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Neural Network Regression Predicted vs Actual')
plt.show()
```


![png](output_44_0.png)



```python
###Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()
```


```python
lasso.fit(X_train, y_train)
```




    Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)




```python
# Score the model
lasso_score = lasso.score(X_test, y_test)
lasso_score
```




    0.9923086310446717




```python
# Score the model
lasso_score = lasso.score(X_train, y_train)
lasso_score
```




    0.9920410049477479




```python
# Make predictions using the testing set
lasso_pred = lasso.predict(X_test)
```


```python
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))

# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lasso_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lasso_pred))
```

    Root mean squared error: 1.65
    Mean absolute error: 0.99
    R-squared: 0.99
    


```python
plt.scatter(y_test, lasso_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Lasso Predicted vs Actual')
plt.show()
```


![png](output_51_0.png)



```python
##ElasticNet
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
```




    ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=None, selection='cyclic', tol=0.0001, warm_start=False)




```python
elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score
```




    0.9923404691889874




```python
elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score
```




    0.9923404691889874




```python
elasticnet_pred = elasticnet.predict(X_test)
```


```python
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, elasticnet_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, elasticnet_pred))
```

    Root mean squared error: 1.64
    Mean absolute error: 0.99
    R-squared: 0.99
    


```python
###Decision Forest Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=200, random_state=1234)
```

    C:\Users\Kingsley\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    


```python
# Train the model using the training sets
regr_rf.fit(X_train, y_train)

```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
               oob_score=False, random_state=1234, verbose=0, warm_start=False)




```python
regr_rf.fit(X_test, y_test)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
               oob_score=False, random_state=1234, verbose=0, warm_start=False)




```python
# Score the model
decision_forest_score = regr_rf.score(X_test, y_test)
decision_forest_score
```




    0.9993148789215262




```python
# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)
```


```python
from math import sqrt
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_rf_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, regr_rf_pred))
```

    Root mean squared error: 0.49
    Mean absolute error: 0.27
    R-squared: 1.00
    


```python
features = X.columns
importances = regr_rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

#Random Forest Regression show how useful they are at predicting the target variable or improve the accuracy of the model
```


![png](output_63_0.png)



```python
plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()
```


![png](output_64_0.png)



```python
#Extra Trees Regression

from sklearn.ensemble import ExtraTreesRegressor
extra_tree = ExtraTreesRegressor(n_estimators=200, random_state=1234)
```


```python
extra_tree.fit(X_train, y_train)
```




    ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None,
              max_features='auto', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
              oob_score=False, random_state=1234, verbose=0, warm_start=False)




```python
extratree_score = extra_tree.score(X_test, y_test)
extratree_score
```




    0.9953993058895532




```python
extratree_score = extra_tree.score(X_train, y_train)
extratree_score
```




    0.9999897264247072




```python
extratree_pred = extra_tree.predict(X_test)
```


```python
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, extratree_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, extratree_pred))
```

    Root mean squared error: 1.27
    Mean absolute error: 0.70
    R-squared: 1.00
    


```python
features = X.columns
importances = extra_tree.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

#Extra Trees Regression show how useful they are at predicting the target variable or improve the accuracy of the model

```


![png](output_71_0.png)



```python
plt.scatter(y_test, extratree_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Extra Trees Predicted vs Actual')
plt.show()
```


![png](output_72_0.png)



```python
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
```

    Scores:
    Linear regression score:  0.9923194367832849
    Neural network regression score:  0.9917593481132061
    Lasso regression score:  0.9920410049477479
    ElasticNet regression score:  0.9923404691889874
    Decision forest score:  0.9993148789215262
    Extra Trees score:  0.9999897264247072
    
    
    RMSE:
    Linear regression RMSE: 1.62
    Neural network RMSE: 1.69
    Lasso RMSE: 1.65
    ElasticNet RMSE: 1.64
    Decision forest RMSE: 0.49
    Extra Trees RMSE: 1.27
    


```python
from IPython.display import Image
Image(filename='mmv.png')
```




![png](output_74_0.png)




```python
import pandas as pd
"""
A framework script that tags the data points according to the gear and assigns it a color and plots the data. 
The gear detection is done by assuming the borders generated using any of the algorithms and placed in
the borders array. 
"""

%matplotlib notebook
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
```

    C:\Users\Kingsley\Anaconda3\lib\site-packages\ipykernel_launcher.py:33: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAJYCAYAAACadoJwAAAgAElEQVR4XuydCbwO1ePGn/e9+45rS1mzREmFJCkh1S+0S3v6S2WpJNm37HuijSItSMoSJVsSEkrKTnayu/t+7/v/nBFd3Hvf5ZyZ98y9z3w+PsU985wz32dmzjx35pzjcLlcLnAjARIgARIgARIgARIgARIgAQsIOBhALKDMKkiABEiABEiABEiABEiABAwCDCA8EUiABEiABEiABEiABEiABCwjwABiGWpWRAIkQAIkQAIkQAIkQAIkwADCc4AESIAESIAESIAESIAESMAyAgwglqFmRSRAAiRAAiRAAiRAAiRAAgwgPAdIgARIgARIgARIgARIgAQsI8AAYhlqVkQCJEACJEACJEACJEACJMAAwnOABEiABEiABEiABEiABEjAMgIMIJahZkUkQAIkQAIkQAIkQAIkQAIMIDwHSIAESIAESIAESIAESIAELCPAAGIZalZEAiRAAiRAAiRAAiRAAiTAAMJzgARIgARIgARIgARIgARIwDICDCCWoWZFJEACJEACJEACJEACJEACDCA8B0iABEiABEiABEiABEiABCwjwABiGWpWRAIkQAIkQAIkQAIkQAIkwADCc4AESIAESIAESIAESIAESMAyAgwglqFmRSRAAiRAAiRAAiRAAiRAAgwgPAdIgARIgARIgARIgARIgAQsI8AAYhlqVkQCJEACJEACJEACJEACJMAAwnOABEiABEiABEiABEiABEjAMgIMIJahZkUkQAIkQAIkQAIkQAIkQAIMIDwHSIAESIAESIAESIAESIAELCPAAGIZalZEAiRAAiRAAiRAAiRAAiTAAMJzgARIgARIgARIgARIgARIwDICDCCWoWZFJEACJEACJEACJEACJEACDCA8B0iABEiABEiABEiABEiABCwjwABiGWpWRAIkQAIkQAIkQAIkQAIkwADCc4AESIAESIAESIAESIAESMAyAgwglqFmRSRAAiRAAiRAAiRAAiRAAgwgPAdIgARIgARIgARIgARIgAQsI8AAYhlqVkQCJEACJEACJEACJEACJMAAwnOABEiABEiABEiABEiABEjAMgIMIJahZkUkQAIkQAIkQAIkQAIkQAIMIDwHSIAESIAESIAESIAESIAELCPAAGIZalZEAiRAAiRAAiRAAiRAAiTAAMJzgARIgARIgARIgARIgARIwDICDCCWoWZFJEACJEACJEACJEACJEACDCA8B0iABEiABEiABEiABEiABCwjwABiGWpWRAIkQAIkQAIkQAIkQAIkwADCc4AESIAESIAESIAESIAESMAyAgwglqFmRSRAAiRAAiRAAiRAAiRAAgwgPAdIgARIgARIgARIgARIgAQsI8AAYhlqVkQCJEACJEACJEACJEACJMAAwnOABEiABEiABEiABEiABEjAMgIMIJahZkUkQAIkQAIkQAIkQAIkQAIMIDwHSIAESIAESIAESIAESIAELCPAAGIZalZEAiRAAiRAAiRAAiRAAiTAAMJzgARIgARIgARIgARIgARIwDICDCCWoWZFJEACJEACJEACJEACJEACDCA8B0iABEiABEiABEiABEiABCwjwABiGWpWRAIkQAIkQAIkQAIkQAIkwADCc4AESIAESIAESIAESIAESMAyAgwglqFmRSRAAiRAAiRAAiRAAiRAAgwgPAdIgARIgARIgARIgARIgAQsI8AAYhlqVkQCJEACJEACJEACJEACJMAAwnOABEiABEiABEiABEiABEjAMgIMIJahZkUkQAIkQAIkQAIkQAIkQAIMIDwHSIAESIAESIAESIAESIAELCPAAGIZalZEAiRAAiRAAiRAAiRAAiTAAMJzgARIgARIgARIgARIgARIwDICDCCWoWZFJEACJEACJEACJEACJEACDCA8B0iABEiABEiABEiABEiABCwjwABiGWpWRAIkQAIkQAIkQAIkQAIkwADCc4AESIAESIAESIAESIAESMAyAgwglqFmRSRAAiRAAiRAAiRAAiRAAgwgPAdIgARIgARIgARIgARIgAQsI8AAYhlqVkQCJEACJEACJEACJEACJMAAwnOABEiABEiABEiABEiABEjAMgIMIJahZkUkQAIkQAIkQAIkQAIkQAIMIDwHSIAESIAESIAESIAESIAELCPAAGIZalZEAiRAAiRAAiRAAiRAAiTAAKLZOeByuZCdnW1JqxwOBwICAoz6RL3c/E+Anvjfg9wtoB96+SFaQ0/08oR+6OUHrxFr/BDPTuLc5+Y7AQYQ39mZsmdWVhZOnDhhivalokFBQShVqhROnjyJzMxMS+pkJQUToCd6nSH0Qy8/RGvoiV6e0A+9/OA1Yo0fpUuXRmBgoDWV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lGVhU1JSlA9keQmNGjVK2f2Wh3ECwlYMx7FjxyC/Wsma+vIRLPuAyAdw2UOPHnrKGu+FcyOMHz8eSUlJXjtw7733KuZbDj08rSrrXMr6Wq5FEz33jZX66euZ80pPnjwZO3bs8NnwZ555Bl26dOE9EiRpterB+yNIgrCaoBCgAQkKZlZCAiRAAiRAAiRAAiRAAiQgBGhAeB2QAAmQAAmQAAmQAAmQAAkEjayFYzkAAAMzSURBVAANSNBQsyISIAESIAESIAESIAESIAEaEF4DJEACJEACJEACJEACJEACQSNAAxI01KyIBEiABEiABEiABEiABEiABoTXAAmQAAmQAAmQAAmQAAmQQNAI0IAEDTUrIgESIAESIAESIAESIAESoAHhNUACJEACJEACJEACJEACJBA0AjQgQUPNikiABEiABEiABEiABEiABGhAeA2QAAmQAAmQAAmQAAmQAAkEjQANSNBQsyISIAESIAESIAESIAESIAEaEF4DJEACJEACJEACJEACJEACQSNAAxI01KyIBEiABEiABEiABEiABEiABoTXAAmQAAmQAAmQAAmQAAmQQNAI0IAEDTUrIgESIAESIAESIAESIAESoAHhNUACJEACJEACJEACJEACJBA0AjQgQUPNikiABEiABEiABEiABEiABGhAeA2QAAmQAAmQAAmQAAmQAAkEjQANSNBQsyISIAESIAESIAESIAESIAEaEF4DJEACJEACJEACJEACJEACQSNAAxI01KyIBEiABEiABEiABEiABEiABoTXAAmQAAmQAAmQAAmQAAmQQNAI0IAEDTUrIgESIAESIAESIAESIAESoAHhNUACJEACJEACJEACJEACJBA0AjQgQUPNikiABEiABEiABEiABEiABGhAeA2QAAmQAAmQAAmQAAmQAAkEjQANSNBQsyISIAESIAESIAESIAESIAEaEF4DJEACJEACJEACJEACJEACQSNAAxI01KyIBEiABEiABEiABEiABEiABoTXAAmQAAmQAAmQAAmQAAmQQNAI0IAEDTUrIgESIAESIAESIAESIAESoAHhNUACJEACJEACJEACJEACJBA0AjQgQUPNikiABEiABEiABEiABEiABGhAeA2QAAmQAAmQAAmQAAmQAAkEjQANSNBQsyISIAESIAESIAESIAESIAEaEF4DJEACJEACJEACJEACJEACQSNAAxI01KyIBEiABEiABEiABEiABEiABoTXAAmQAAmQAAmQAAmQAAmQQNAI0IAEDTUrIgESIAESIAESIAESIAESoAHhNUACJEACJEACJEACJEACJBA0Av8fDaL5KwBpJNoAAAAASUVORK5CYII=" width="640">





    []


