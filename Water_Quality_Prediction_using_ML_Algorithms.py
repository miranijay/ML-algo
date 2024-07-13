#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np                  # Numerical Python, performing mathematical and logical operations on Arrays. 
import pandas as pd                 # Importing data, manipulation of tabular data in DataFrames.
import seaborn as sns               # Plotting functions operate on dataframes, helps you explore and understand your data.
import matplotlib.pyplot as plt     # To create static and animated visualisations.
get_ipython().run_line_magic('matplotlib', 'inline                  # turns on inline plotting, where plot graphics will appear.')
import plotly.express as px         # contains functions that can create entire figures at once.


# In[2]:


df = pd.read_csv('water_potability.csv')       # Importing the csv file.


# In[3]:


df.head()               # Returns the first 5 rows(default) of the dataframe.


# # Following are the list of algorithms that are used :-
#  
#     Logistic Regression
#     Decision Tree
#     Random Forest
#     KNeighbours
#     SVM
#     AdaBoost

# In[4]:


df.shape           # tells the number of rows and columns of a given DataFrame.


# In[5]:


df.columns         # It returns name of all the columns present in Dataframe.


# In[6]:


df.describe()    # Returns description of the data present in the DataFrame.


# In[7]:


df.info()        # prints information about the DataFrame.


# In[8]:


df.nunique()        # returns the number of unique values for each column.


# In[9]:


df.isnull().sum()     #returns the sum of number of missing values in the dataset.


# In[10]:


df.dtypes             # returns a data type of each column.


# In[11]:


plt.figure(figsize=(10,8))
sns.heatmap(df.isnull())      # graphical representations of data, generalized view of numeric values.


# In[12]:


df.corr()             #relationship between two variables, positive sign indicates directly proportional, negative sign indicates inversely proportional


# In[13]:


plt.figure(figsize=(10,8))                         #width and height of the figure in inches.
sns.heatmap(df.corr(), annot=True, cmap='cool')    #plot that visualize the strength of relationships between numerical variables.


# In[14]:


# Unstacking the correlation matrix to see the values more clearly.

df.corr().abs().unstack().sort_values()            


# In[15]:


plt.figure(figsize=(10,8))
sns.countplot(x='Potability', data=df)           #Show the counts of Potability(0,1) in graphical presentation.


# In[16]:


df.Potability.value_counts()       # Show the count of Potability in numeric value.


# In[17]:


# A violin plot is more informative than a plain box plot. 
# While a box plot only shows summary statistics such as mean/median and interquartile ranges, 
# the violin plot shows the full distribution of the data. 


# In[18]:


plt.figure(figsize=(10,8))
sns.violinplot(x='Potability', y='ph', data=df, palette='rocket')    # used to visualize the distribution of numerical data.


# In[19]:


# Visualizing dataset and also checking for outliners.

fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(10,20))          #describes the layout of the figure.
index=0
ax=ax.flatten()                                                    #returns a flat(one-dimensional) version of the array.

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index+=1

plt.tight_layout()                                                 #automatically adjusts subplot params so that the subplot(s) fits in to the figure area


# In[20]:


plt.rcParams['figure.figsize']=[20,10]               #defines a runtime configuration(rc).
df.hist()                  #quick way to understand the distribution of certain numerical variables from the dataset.


# In[21]:


sns.pairplot(df, hue='Potability')       #allows us to plot pairwise relationships between variables within a dataset.


# In[22]:


plt.figure(figsize=(10,8)) 
sns.distplot(df['Potability'])            #represents the overall distribution of continuous data variables.


# In[23]:


df.hist(column='ph', by='Potability')


# In[24]:


df.hist(column='Hardness', by='Potability')  #quick way to understand the distribution of certain numerical variables from the dataset.
  


# In[25]:


df.hist(column='Solids', by='Potability')


# In[26]:


sns.histplot(x='Hardness', data=df)        #Plot univariate or bivariate histograms to show distributions of datasets.


# In[27]:


sns.histplot(x='Potability',y='ph', data=df)   #classic visualization tool 


# In[28]:


df.skew().sort_values()


# In[29]:


# Using pandas skew function to check the correlation between the values.
# Values between 0.5 to -0.5 will be considered as the normal distribution 
# else will be skewed depending upon the skewness value.


# In[30]:


import plotly.express as px
#plotly is a library consisting of a collection of 40 different types of graphs
#-px function is used to plot column graphs which are easy to use


# In[31]:


fig = px.box(df, x="Potability", y="ph", color="Potability", width=800, height=400)
fig.show()


# In[32]:


fig = px.box(df, x="Potability", y="Hardness", color="Potability", width=800, height=400)
fig.show()


# In[33]:


fig = px.histogram (df, x = "Sulfate",  facet_row = "Potability",  template = 'plotly_dark')
fig.show ()


# In[34]:


fig = px.histogram (df, x = "Trihalomethanes",  facet_row = "Potability",  template = 'plotly_dark')
fig.show ()


# In[35]:


fig =  px.pie(df, names = "Potability", hole = 0.4, template = "plotly_dark")
fig.show ()


# In[36]:


fig = px.scatter (df, x = "ph", y = "Sulfate", color = "Potability", template = "plotly_dark",  trendline="ols")
fig.show ()


# In[37]:


fig = px.scatter (df, x = "Organic_carbon", y = "Hardness", color = "Potability", template = "plotly_dark",  trendline="lowess")
fig.show ()


# In[38]:


df.isnull().mean().plot.bar(figsize=(10,6)) 
plt.ylabel('Percentage of missing values') 
plt.xlabel('Features') 
plt.title('Missing Data in Percentages');


# In[39]:


#insull() method replaces all the values with a boolean value


# In[40]:


#here fillna is filling all the null values with a specified value
df['ph'] = df['ph'].fillna(df['ph'].mean())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())


# In[114]:


print(df.head())


# In[42]:


plt.figure(figsize=(10,8))
sns.heatmap(df.isnull())


# In[43]:


df.isnull().sum()


# In[44]:


#here we are specifying which specific row or column to be removed
X = df.drop('Potability', axis=1)
y = df['Potability']


# In[45]:


X.shape, y.shape


# In[46]:


# import StandardScaler to perform scaling which removes mean and scales each variable as a single unit
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()


# In[47]:


#fit_transform is useed too train the data so as to learn the scaling parameters if the data
X = scaler.fit_transform(X)
X


# In[48]:


# import train-test split 
from sklearn.model_selection import train_test_split


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#  # Using Decision Tree

# It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.

# In[50]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[108]:


import joblib


# In[51]:


from sklearn.tree import DecisionTreeClassifier


# In[52]:


# Creating Model Object

model_dt = DecisionTreeClassifier(max_depth=7, random_state=42)


# In[53]:


# Train the model

model_dt.fit(X_train, y_train)


# In[54]:


# Making Predictions

pred_dt = model_dt.predict(X_test)


# In[55]:


# Calculating the accuracy

dt = accuracy_score(y_test, pred_dt)
print(dt)


# In[56]:


print(classification_report(y_test, pred_dt))


# In[57]:


# Confusion Matrix

cm2 = confusion_matrix(y_test, pred_dt)
cm2


# In[58]:


plt.figure(figsize=(10,8))
sns.heatmap(cm2/np.sum(cm2), annot=True, fmt='0.2%', cmap='Reds')


# # Using Random Forest

# It is constructed by using multiple decision trees and the final decision is obtained by majority votes of the decision tree.

# In[59]:


from sklearn.ensemble import RandomForestClassifier


# In[60]:


# Creating Model Object

model_rf = RandomForestClassifier(n_estimators=900,criterion='gini', random_state=42)


# In[61]:


# Training Model

model_rf.fit(X_train, y_train)


# In[62]:


# Making Prediction

pred_rf = model_rf.predict(X_test)


# In[63]:


# Calculating Accuracy Score

rf = accuracy_score(y_test, pred_rf)
print(rf)


# In[64]:


print(classification_report(y_test,pred_rf))


# In[65]:


# confusion Maxtrix

cm3 = confusion_matrix(y_test, pred_rf)
print(cm3)
plt.figure(figsize=(10,8))
sns.heatmap(cm3/np.sum(cm3), annot = True, fmt=  '0.2%', cmap = 'Reds')


# Why random forest is better than decision tree ?
# 
# ##if the size of the datasets is huge then in that case one single decision tree would lead to a overfitting.
# 

# # Using K-Nearest Neighbors

# * It classifies a datapoint(testing data) based on how it neighbors are classified.
# * A datapoint is classified by majority votes from its nearest neighbors. 

# In[66]:


from sklearn.neighbors import KNeighborsClassifier


# In[67]:


import math
print(math.sqrt(len(y_train)))
print(math.sqrt(len(y_test)))


# In[68]:


# Creating Model Object

model_kn = KNeighborsClassifier(n_neighbors=39, p=2, metric='euclidean')


# In[69]:


#Train model

model_kn.fit(X_train, y_train)


# In[70]:


#Predict the test set results

pred_kn = model_kn.predict(X_test)


# In[71]:


pred_kn


# In[72]:


kn = accuracy_score(y_test, pred_kn)
kn


# In[73]:


print(classification_report(y_test, pred_kn))


# In[74]:


# Confusion Matrix

cm4 = confusion_matrix(y_test, pred_kn)
cm4


# In[75]:


plt.figure(figsize=(10,8))
sns.heatmap(cm4/np.sum(cm4), annot=True, fmt='0.2%', cmap='coolwarm')


# # Using SVM

# Support vector machines (SVMs) are supervised machine learning algorithms for outlier detection, regression, and classification. it basically helps us to detect the values that are abnormal in the data.

# In the svm model what we do is that we input a random state and keep it's value constant so that it would provide us the same values in all the algorithms

# In[76]:


from sklearn.svm import SVC


# In[77]:


# Creating Model Object

model_svm = SVC(kernel='rbf', random_state = 42)


# In[78]:


#Train model

model_svm.fit(X_train, y_train)


# In[79]:


# Making Prediction

pred_svm = model_svm.predict(X_test)


# In[123]:


pred_svm


# In[80]:


# Calculating Accuracy Score

sv = accuracy_score(y_test, pred_svm)
print(sv)


# In[81]:


print(classification_report(y_test,pred_svm))


# In[82]:


# confusion Maxtrix

cm5 = confusion_matrix(y_test, pred_svm)
print(cm5)

plt.figure(figsize=(10,8))
sns.heatmap(cm5/np.sum(cm5), annot = True, fmt=  '0.2%', cmap = 'Reds')


# # Using AdaBoost Classifier

# The basic concept behind Adaboost is to set the weights of classifiers and training the data sample in each iteration such that it ensures the accurate predictions of unusual observations.

# In[83]:


from sklearn.ensemble import AdaBoostClassifier


# In[84]:


# Creating Model Object

model_ada = AdaBoostClassifier(learning_rate= 0.002,n_estimators= 205,random_state=42)


# In[85]:


#Train model

model_ada.fit(X_train, y_train)


# In[86]:


# Making Prediction

pred_ada = model_ada.predict(X_test)


# In[87]:


# Calculating Accuracy Score

ada = accuracy_score(y_test, pred_ada)
print(ada)


# In[88]:


print(classification_report(y_test,pred_ada))


# In[89]:


# confusion Maxtrix

cm5 = confusion_matrix(y_test, pred_ada)
print(cm5)

plt.figure(figsize=(10,8))
sns.heatmap(cm5/np.sum(cm5), annot = True, fmt=  '0.2%', cmap = 'Reds')


# # Using Logistic Regression 

# * Logistic regression predicts the output of a categorical dependent variable.
# * Logistic regression is one of the most popular algorithms for binary classification. Given a set of examples with features, the goal of logistic regression is to output values between 0 and 1.

# In[90]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[91]:


# Creating model object

model_lg = LogisticRegression(max_iter=120,random_state=0, n_jobs=20)


# In[92]:


# Training Model

model_lg.fit(X_train, y_train)


# In[93]:


# Making Prediction

pred_lg = model_lg.predict(X_test)


# In[94]:


# Calculating Accuracy Score

lg = accuracy_score(y_test, pred_lg)
print(lg)


# In[95]:


print(classification_report(y_test,pred_lg))


# In[96]:


# confusion Maxtrix

cm6 = confusion_matrix(y_test, pred_lg)
cm6


# In[97]:


plt.figure(figsize=(10,8))
sns.heatmap(cm6/np.sum(cm6), annot = True, fmt=  '0.2%', cmap = 'Reds')


# # Comparision between Algorithms

# In[98]:


models = pd.DataFrame({
    'Model':['Decision Tree', 'Random Forest', 'KNeighbours', 'SVM', 'AdaBoost', 'Logistic Regression'],
    'Accuracy_score' :[dt, rf, kn, sv, ada, lg]
})
models
sns.barplot(x='Accuracy_score', y='Model', data=models)

models.sort_values(by='Accuracy_score', ascending=False)


# In[142]:


joblib.dump(model_kn,"knn")


# In[143]:


abc = joblib.load("knn")


# In[144]:


abc.predict([[7.080795, 204.890455, 20791.318981, 7.300212, 368.516441, 564.308654,10.379783, 86.990970, 2.963135]])


# In[ ]:





# In[119]:


joblib.dump(model_svm,"svm")


# In[121]:


xyz = joblib.load("svm")


# In[122]:


xyz.predict([[7.080795, 204.890455, 20791.318981, 7.300212, 368.516441, 564.308654,10.379783, 86.990970, 2.963135]])


# In[ ]:





# In[124]:


joblib.dump(model_dt,"dt")


# In[125]:


mno = joblib.load("dt")


# In[126]:


mno.predict([[7.080795, 204.890455, 20791.318981, 7.300212, 368.516441, 564.308654,10.379783, 86.990970, 2.963135]])


# In[ ]:





# In[132]:


joblib.dump(model_lg,"lg")


# In[133]:


qr = joblib.load("lg")


# In[134]:


qr.predict([[7.080795, 204.890455, 20791.318981, 7.300212, 368.516441, 564.308654,10.379783, 86.990970, 2.963135]])


# In[ ]:





# In[135]:


joblib.dump(model_lg,"ada")


# In[137]:


rs = joblib.load("ada")


# In[138]:


rs.predict([[7.080795, 204.890455, 20791.318981, 7.300212, 368.516441, 564.308654,10.379783, 86.990970, 2.963135]])


# In[ ]:





# In[139]:


joblib.dump(model_rf,"rf")


# In[140]:


ran = joblib.load("ada")


# In[141]:


rs.predict([[7.080795, 204.890455, 20791.318981, 7.300212, 368.516441, 564.308654,10.379783, 86.990970, 2.963135]])


# In[ ]:





# In[ ]:





# In[99]:


print(df.columns)


# In[102]:


def water(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,Organic_carbon, Trihalomethanes, Turbidity) :
    #turning the agruments into a numpy array
    
    x = np.array(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,Organic_carbon, Trihalomethanes, Turbidity)
    
    pred = model_kn.predict(x.reshape(1,-1))
    
    return pred


# In[103]:


import gradio as gr


# In[104]:


outputs = gr.outputs.Textbox()

app = gr.Interface(fn=water, inputs=['number','number','number','number','number','number','number','number','number'], outputs=outputs,description="This is a water Quality model")


# In[105]:


app.launch()


# In[106]:


app.launch(share=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




