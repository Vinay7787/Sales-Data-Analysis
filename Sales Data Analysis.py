#!/usr/bin/env python
# coding: utf-8

# ##### ABC Private Limited, a retail company, wants to gain insight into its customers' purchasing habits, particularly their spending on different product categories. To do this, they have provided a summary of the purchase history of a number of high-volume products from the previous month. This data includes information on the customer demographics, such as age, gender, marital status, city type, and length of stay in their current city, as well as details on the products themselves, including product ID and product category.

# In[95]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[96]:


data=pd.read_csv("D:/train.csv")


# In[97]:


data


# In[98]:


data.info() #information About dataset


# In[99]:


print(data.describe(include=[object])) #find the correlation between each colounm


# In[100]:


print(data.describe(include=[int]))


# In[101]:


print(data.describe(include=[float]))


# In[102]:


data.shape #retrive the dimension of object


# In[103]:


data.nunique() #unique values in the data


# In[104]:


data.isna()


# In[105]:


data.duplicated().sum() #Check the duplicate's  value


# In[106]:


data.isnull().sum() / data.shape[0] * 100       #Check the percentages of missing value


# In[107]:


data.Age.value_counts() #check the value counts


# In[108]:


sns.displot(data, x="Age") #Age Count


# In[109]:


d1= sns.boxplot(x=data["Purchase"]) #outlier in purchase


# In[110]:


d2= sns.boxplot(x=data["Occupation"]) #find outlier in occupation


# In[111]:


cor = data.corr()


# In[112]:


sns.heatmap(cor, annot = True, cmap= 'coolwarm')  #correlation between variables in the data


# In[113]:


print(data['Purchase'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(data['Purchase'], color='b', bins=100, hist_kws={'alpha': 0.4}); #Find the purchase density using distplot


# In[114]:


gender_map={'Gender':{"M":0,"F":1}}


# In[118]:


data = data.replace(gender_map) #Convert categorical data into integer using map function 


# In[116]:


data


# In[119]:


print(data.isna().sum()) #missing value treatment


# In[121]:


data.columns=['User_ID','Product_ID','Gender','Age','Occupation','City_Category','SICCY','MS','PC1','PC2','PC3','Purchase'] #Rename columns


# In[122]:


data


# In[124]:


data[['PC2', 'PC3']] = data[['PC2', 'PC3']].fillna(data[['PC2', 'PC3']].mean())  #fill nan values


# In[128]:


data.to_excel("D:/train.xlsx",header=True,index=False)


# In[130]:


data


# In[ ]:


Age1= [data['Age'].between(0,17), data['Age'].between(18,25), data['Age'].between(26, 35),data['Age'].between(36, 45),data['Age'].between(46,50),data['Age'].between(51,55),data['Age'].between(56,100)]
values = [1, 2, 3,4,5,6,7]
                                                #map range variables into integers (e.g 'Age' column)
data['b'] = np.select(Age1, values, 0)


# # Question:2

# In[155]:


from sklearn.model_selection import train_test_split
from sklearn import tree
clf = tree.DecisionTreeClassifier()


# In[160]:


x=data['Occupation']
y=data['Purchase']


# In[161]:


x_train, X_test, y_train, y_test = train_test_split(
                      x, y, test_, random_state=42)


# In[ ]:


clf.fit(x_train, y_train) 
print(clf.score(X_test, y_test))

