#!/usr/bin/env python
# coding: utf-8

# # Importing the data

# In[1]:


#To import the data set
import pandas as pd
breast_cancer = pd.read_csv(("//pepper/homestd/My Documents/Doctorado Bristol/Data science/Assesment/METABRIC_RNA_Mutation.csv"), low_memory=False)


# In[2]:


#To visualize the dataset
breast_cancer


# # Linear regression model to determine influence of age in the pacient survival

# In[166]:


#to import the linear regression model from scikit learn
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)


# In[167]:


#To define X and Y variables
x = breast_cancer[["age_at_diagnosis"]]
y = breast_cancer["overall_survival_months"]


# In[168]:


#to set up the model
model.fit(x, y)


# In[169]:


x_fit = pd.DataFrame({"x": [0, 100]})


# In[170]:


y_pred = model.predict(x_fit)


# In[171]:


import matplotlib.pyplot as plt


# In[172]:


#To plot the linear regression model results
fig, ax = plt.subplots()
breast_cancer.plot.scatter("age_at_diagnosis", "overall_survival_months", ax=ax)
ax.plot(x_fit["x"], y_pred, linestyle=":")


# In[10]:


#To print the gradient and the intercept of the model
print(" Model gradient: ", model.coef_[0])
print("Model intercept:", model.intercept_)


# # Testing the regression linear model done with patient age

# In[11]:


#note: In this case it does not make so much sense to test the regression linear model because clearly there is no linear relation, however with other features or other datasets could be useful to do this test


# In[12]:


#to divide the dataset in train and test data
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)


# In[13]:


#to plot the results of the model test
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(train_x, train_y, color="red", marker="o", label="train")
ax.scatter(test_x, test_y, color="blue", marker="x", label="test")
ax.legend()


# In[14]:


from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(train_x, train_y)


# In[15]:


#to calculate the model score
model.score(test_x, test_y)


# # Linear regression model to determine influence of tumor size in the pacient survival

# In[16]:


#to import the linear regression model from scikit learn
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)


# In[17]:


#To define X and Y variables
x = breast_cancer[["tumor_size"]]
y = breast_cancer["overall_survival_months"]


# In[18]:


#to set up the model
model.fit(x, y)


# In[19]:


x_fit = pd.DataFrame({"x": [0, 180]})


# In[20]:


y_pred = model.predict(x_fit)


# In[21]:


#To plot the linear regression model results
fig, ax = plt.subplots()
breast_cancer.plot.scatter("tumor_size", "overall_survival_months", ax=ax)
ax.plot(x_fit["x"], y_pred, linestyle=":")


# In[22]:


#To print the gradient and the intercept of the model
print(" Model gradient: ", model.coef_[0])
print("Model intercept:", model.intercept_)


# # Determining the main factors afecting survival

# In[23]:


#to be able to see the complete correlation table
pd.set_option("display.max_columns", 520)
pd.set_option("display.max_rows", 520)


# In[26]:


#determining the correlation among features
Breast_cancer_corr = breast_cancer.corr()
Breast_cancer_corr


# In[28]:


#To see only the correlations of the variable of interest
overall_survival_corr = Breast_cancer_corr["overall_survival"]
overall_survival_corr


# In[30]:


#to sort the correlation dataframe and be able to choose more correlated variables for subsequent analysis
sorted_over_corr = overall_survival_corr.sort_values()
print(sorted_over_corr)


# In[84]:


#To visualize the first 12 lines (strong positive correlation with overall survival)
sorted_over_corr_head = sorted_over_corr.head(12)
print(sorted_over_corr_head)


# # Decision tree model to classify individuals features in propensity to live or die

# In[125]:


#Load of the requiered packages
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 


# In[126]:


#to prepare the dataset
import pandas as pd
import numpy as np

def clean_dataset(breast_cancer):
    assert isinstance(breast_cancer, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~breast_cancer.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# In[143]:


#to define variables
feature_cols = ['age_at_diagnosis', 'gsk3b', 'tumor_size', 'tumor_stage', 'kmt2c', 'lymph_nodes_examined_positive', 'mutation_count', 'tnk2'] 
x = breast_cancer[feature_cols]
y = breast_cancer['overall_survival']


# In[144]:


#split data set 70% training and 30% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[145]:


#create the decision tree object
clf = DecisionTreeClassifier()


# In[146]:


x = x.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)


# In[147]:


breast_cancer = breast_cancer.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


# In[148]:


#to train the decision tree
clf = clf.fit(x_train, y_train) 


# In[149]:


#to predict the response for test dataset
y_pred = clf.predict(x_test) 


# # Evaluation the model by the accuracy estimate

# In[150]:


#to estimate the model accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[151]:


#to import sklearn packages
import six
import sys
sys.modules['sklearn.externals.six'] = six


# In[152]:


#to load requiered packages for visualization of the decision tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus


# In[153]:


conda install -c conda-forge pydotplus


# In[154]:


conda update -n base -c defaults conda


# In[155]:


#to obtain the decision tree figure
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('cancer_breast.png')
Image(graph.create_png())


# # Optimizing decision tree

# In[156]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)


# In[157]:


# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)


# In[158]:


#Predict the response for test dataset
y_pred = clf.predict(x_test)


# In[159]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[160]:


#To obtain the decision tree figure
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree2.png')
Image(graph.create_png())

