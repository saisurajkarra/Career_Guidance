#!/usr/bin/env python
# coding: utf-8

# # Importing All the Necessary Libraries

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea


# # Loading the CSV file

# In[2]:


data = pd.read_csv("data.csv")


# # Viewing the starting 5 Rows and Columns

# In[3]:


data.head()


# # Viewing Suggested Job Role which is the final prediction to be done

# In[4]:


data["Suggested Job Role"].describe()
    


# # To check all the labels 

# In[5]:


data.columns


# # Describe one of the data to know its data type and size

# In[6]:


os=data["Acedamic percentage in Operating Systems"]


# In[7]:


os.describe


# In[8]:


bp = data.boxplot("Acedamic percentage in Operating Systems")
plt.show()


# # Plotting Graphs

# In[32]:


bp = data.boxplot("percentage in Algorithms")
plt.show()


# In[33]:


bp = data.boxplot("Percentage in Programming Concepts")
plt.show()


# In[34]:


bp = data.boxplot("Percentage in Software Engineering")
plt.show()


# In[35]:


bp = data.boxplot("Percentage in Computer Networks")
plt.show()


# In[36]:


bp = data.boxplot("Percentage in Electronics Subjects")
plt.show()


# In[37]:


bp = data.boxplot("Percentage in Computer Architecture")
plt.show()


# In[38]:


bp = data.boxplot("Percentage in Mathematics")
plt.show()


# In[39]:


bp = data.boxplot("Percentage in Communication skills")
plt.show()


# In[40]:


bp = data.boxplot("Hours working per day")
plt.show()


# In[41]:


sea.pairplot(data=data)


# In[42]:


bp = data.boxplot("Logical quotient rating")
plt.show()


# In[43]:



bp = data.boxplot("hackathons")
plt.show()


# In[194]:


bp = data.boxplot("coding skills rating")
plt.show()


# In[195]:


bp = sea.barplot(y="public speaking points", data=data)[:100]


# In[ ]:


sea.countplot(x='public speaking points',data=data)


# In[196]:


print(pd.pivot_table(data=data,index='can work long time before system?',aggfunc='size'))
sea.countplot(x='can work long time before system?',data=data)


# In[1]:


print(pd.pivot_table(data=data,index='self-learning capability?',aggfunc='size'))
sea.countplot(x='self-learning capability?',data=data)


# In[20]:


print(pd.pivot_table(data=data,index='Extra-courses did',aggfunc='size'))
sea.countplot(x='Extra-courses did',data=data)


# In[21]:


print(pd.pivot_table(data=data,index='certifications',aggfunc='size'))
sea.countplot(y='certifications',data=data)


# In[22]:


print(pd.pivot_table(data=data,index='Extra-courses did',aggfunc='size'))
sea.countplot(x='Extra-courses did',data=data)


# In[23]:



print(pd.pivot_table(data=data,index='workshops',aggfunc='size'))
sea.countplot(y='workshops',data=data)


# In[24]:



print(pd.pivot_table(data=data,index='talenttests taken?',aggfunc='size'))
sea.countplot(x='talenttests taken?',data=data)


# In[25]:



print(pd.pivot_table(data=data,index='olympiads',aggfunc='size'))
sea.countplot(x='olympiads',data=data)


# In[26]:


print(pd.pivot_table(data=data,index='reading and writing skills',aggfunc='size'))
sea.countplot(x='reading and writing skills',data=data)


# In[27]:


print(pd.pivot_table(data=data,index='memory capability score',aggfunc='size'))
sea.countplot(x='memory capability score',data=data)


# In[28]:


print(pd.pivot_table(data=data,index='Interested subjects',aggfunc='size'))
sea.countplot(y='Interested subjects',data=data)


# In[29]:


print(pd.pivot_table(data=data,index='interested career area ',aggfunc='size'))
sea.countplot(y='interested career area ',data=data)


# In[30]:


print(pd.pivot_table(data=data,index='Job/Higher Studies?',aggfunc='size'))
sea.countplot(x='Job/Higher Studies?',data=data)


# In[31]:


print(pd.pivot_table(data=data,index='Type of company want to settle in?',aggfunc='size'))
sea.countplot(y='Type of company want to settle in?',data=data)


# In[32]:


print(pd.pivot_table(data=data,index='Taken inputs from seniors or elders',aggfunc='size'))
sea.countplot(x='Taken inputs from seniors or elders',data=data)


# In[33]:


print(pd.pivot_table(data=data,index='interested in games',aggfunc='size'))
sea.countplot(x='interested in games',data=data)


# In[34]:


print(pd.pivot_table(data=data,index='Interested Type of Books',aggfunc='size'))
sea.countplot(y='Interested Type of Books',data=data)


# In[35]:


print(pd.pivot_table(data=data,index='Salary Range Expected',aggfunc='size'))
sea.countplot(x='Salary Range Expected',data=data)


# In[36]:



print(pd.pivot_table(data=data,index='In a Realtionship?',aggfunc='size'))
sea.countplot(x='In a Realtionship?',data=data)


# In[37]:


print(pd.pivot_table(data=data,index='Gentle or Tuff behaviour?',aggfunc='size'))
sea.countplot(x='Gentle or Tuff behaviour?',data=data)


# In[38]:


print(pd.pivot_table(data=data,index='Management or Technical',aggfunc='size'))
sea.countplot(x='Management or Technical',data=data)


# In[39]:


print(pd.pivot_table(data=data,index='Salary/work',aggfunc='size'))
sea.countplot(x='Salary/work',data=data)


# In[40]:


print(pd.pivot_table(data=data,index='hard/smart worker',aggfunc='size'))
sea.countplot(x='hard/smart worker',data=data)


# In[41]:


print(pd.pivot_table(data=data,index='worked in teams ever?',aggfunc='size'))
sea.countplot(x='worked in teams ever?',data=data)


# In[42]:


print(pd.pivot_table(data=data,index='Introvert',aggfunc='size'))
sea.countplot(x='Introvert',data=data)


# In[284]:


print(pd.pivot_table(data=data,index='Suggested Job Role',aggfunc='size'))
sea.countplot(y='Suggested Job Role',data=data)


# # Now we Scale the data using Standard Scaler 

# z = (x - u) / s
# 
# where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

# In[35]:


from sklearn.preprocessing import StandardScaler


# In[36]:


pos=data.iloc[:,:1]


# In[37]:


pos.describe()


# In[38]:


from sklearn.preprocessing import MinMaxScaler


# In[39]:


sc = MinMaxScaler(feature_range=(0,3))


# In[40]:


t1 = sc.fit_transform(pos)


# In[41]:


t1


# In[42]:


len(t1)


# # Now we do Label Encoding to the given Data

# In[43]:


from sklearn.preprocessing import LabelEncoder


# In[44]:


pos=data.iloc[:1,:]


# In[45]:


pos


# In[46]:


le = LabelEncoder()


# In[47]:


data['In a Realtionship?']


# In[48]:


from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
data['In a Realtionship?']= label_encoder.fit_transform(data['In a Realtionship?']) 
  
data['In a Realtionship?'].unique() 


# In[49]:


data['In a Realtionship?']


# In[50]:



# Encode labels in column 'species'. 
data['Interested Type of Books']= label_encoder.fit_transform(data['Interested Type of Books']) 
  
data['Interested Type of Books'].unique()


# In[51]:


data["Interested Type of Books"]


# In[52]:


data.columns


# In[53]:


columns=data.columns


# # Now we apply MinMax Scaler
# 

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
# 

# In[54]:


sc = MinMaxScaler(feature_range=(0,1))


# In[56]:


ndata=data.iloc[:,:14]
ndata


# In[57]:


ndata=sc.fit_transform(ndata)


# In[58]:


ndata


# In[59]:


ndata=sc.fit_transform(ndata)


# In[60]:


cdata=data.iloc[:,14:39]


# In[62]:


cdata


# In[63]:


data1=data.iloc[:,14:39].values


# In[64]:


for i in range (25):
    data1[:,i]=label_encoder.fit_transform(data1[:,i])


# In[65]:


type(data1)


# In[66]:


data1


# In[67]:


ndata.shape


# In[68]:


data1.shape


# In[69]:


fdata = np.append(ndata,data1,axis=1)


# In[70]:


fdata


# In[71]:


fdata.shape


# In[72]:


dvar=fdata[:,:-1]


# In[73]:


dvar


# In[74]:


indvar=fdata[:,-1]


# In[75]:


indvar


# In[76]:


datatotal=pd.DataFrame(fdata)


# In[77]:


datatotal


# In[78]:


data.columns


# In[79]:


datatotal.columns=(['Acedamic percentage in Operating Systems', 'percentage in Algorithms',
       'Percentage in Programming Concepts',
       'Percentage in Software Engineering', 'Percentage in Computer Networks',
       'Percentage in Electronics Subjects',
       'Percentage in Computer Architecture', 'Percentage in Mathematics',
       'Percentage in Communication skills', 'Hours working per day',
       'Logical quotient rating', 'hackathons', 'coding skills rating',
       'public speaking points', 'can work long time before system?',
       'self-learning capability?', 'Extra-courses did', 'certifications',
       'workshops', 'talenttests taken?', 'olympiads',
       'reading and writing skills', 'memory capability score',
       'Interested subjects', 'interested career area ', 'Job/Higher Studies?',
       'Type of company want to settle in?',
       'Taken inputs from seniors or elders', 'interested in games',
       'Interested Type of Books', 'Salary Range Expected',
       'In a Realtionship?', 'Gentle or Tuff behaviour?',
       'Management or Technical', 'Salary/work', 'hard/smart worker',
       'worked in teams ever?', 'Introvert', 'Suggested Job Role'])


# In[80]:


datatotal


# In[81]:


x=datatotal.iloc[:,:-1].values


# In[82]:


x


# In[83]:


y=datatotal.iloc[:,-1].values


# In[84]:


y


# # Now We split data into 70:30 

# In[85]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)


# In[86]:


len(X_train)


# In[87]:


X_train.shape


# In[88]:


X_train.size


# In[89]:


lab_enc = preprocessing.LabelEncoder()
y_trainn = lab_enc.fit_transform(y_train)


# # Applying Logistic Regression

# In[90]:


from sklearn.linear_model import LogisticRegression


# In[91]:


regression = LogisticRegression()


# In[92]:


regression.fit(X_train,y_trainn)


# In[93]:


X_train


# In[94]:


pred = regression.predict(X_test)


# In[95]:


pred


# In[96]:


lab_enc = preprocessing.LabelEncoder()
y_test = lab_enc.fit_transform(y_test)


# # We run Classification Report

# In[108]:


from sklearn.metrics import *


# In[109]:


print(classification_report(y_test,pred))


# # Creating the Confusion Matrix

# In[110]:


con = confusion_matrix(y_test,pred)


# In[111]:


y_test


# In[112]:


pred


# In[113]:


print(con)


# In[114]:


acc = accuracy_score(pred,y_test)


# In[115]:


acc*100


# # Importing libraries for Decision tree

# In[116]:


from sklearn import tree


# In[117]:


import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor,RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn import tree
from sklearn.metrics import accuracy_score
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
pd.set_option('display.notebook_repr_html', False)
#get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-white')
print("Package Loaded")


# In[118]:



from sklearn import metrics

model = RandomForestClassifier(max_depth=10,n_estimators=500)#DecisionTreeClassifier()

model.fit(X_train,y_trainn)


# In[119]:


prediction = model.predict(X_test)
prediction


# In[120]:


expected = y_test
predicted = prediction
conf = metrics.confusion_matrix(expected, predicted)
print(conf)


# In[121]:


from sklearn.svm import SVC
model = SVC()

model.fit(X_train,y_trainn)


# In[122]:


predictions = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[123]:


acc = accuracy_score(predictions,y_test)


# In[124]:


acc


# In[125]:


from xgboost import XGBClassifier


# In[148]:


clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 10)
clf_entropy.fit(X_train, y_trainn)


# In[149]:


entropy_y_pred=clf_entropy.predict(X_test)


# In[150]:


cm_entopy = confusion_matrix(y_test,entropy_y_pred)


# In[152]:


print("confusion matrics=",cm_entopy)
print("  ")
print("accuracy=",entropy_accuracy)

entropy_accuracy = accuracy_score(y_test,entropy_y_pred)


# In[165]:


clf_entropy = tree.DecisionTreeClassifier(criterion = "gini", random_state = 10)
clf_entropy.fit(X_train, y_trainn)


# In[166]:


entropy_y_pre=clf_entropy.predict(X_test)


# In[167]:


cm_entop = confusion_matrix(y_test,entropy_y_pre)


# In[168]:


print("confusion matrics=",cm_entop)
print("  ")
entropy_accuracy = accuracy_score(y_test,entropy_y_pre)
print("accuracy=",entropy_accuracy)

