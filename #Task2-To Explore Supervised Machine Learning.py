#!/usr/bin/env python
# coding: utf-8
Linear Regression Model
 In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables. 
# In[2]:


#Importing all libraries

import pandas as pd
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Reading data from external link

url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head()


# In[4]:


#Plotting the distribution of scores
data.plot(x='Hours', y='Scores',style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From above graph, we can clearly see that there is a positive linear relation between the no of hours studied and percentage of score.

# In[5]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[6]:


#Splitting data into training set and testing set
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=2020) 


# In[7]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# In[8]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line)
plt.show()


# In[9]:


y_pred = regressor.predict(X_test)
y_pred


# In[10]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


# In[11]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[12]:


# test with data
hr = 9.25
hrs=np.array(hr).reshape(1,-1)
hr_pred = regressor.predict(hrs)
print("No of Hours = {}".format(hrs))
print("Predicted Score = {}".format(hr_pred[0]))


# In[ ]:




