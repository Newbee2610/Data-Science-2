#!/usr/bin/env python
# coding: utf-8

# # PROJECT-1 FAKE NEWS DETECTION USING PYTHON

# It is very much easy to spread all the fake information in today’s all-connected world across the internet. Fake news is sometimes transmitted through the internet by some unauthorised sources, which creates issues for the targeted person and it makes them panic and leads to even violence. 
# To combat the spread of fake news, it’s critical to determine the information’s legitimacy, which this Data Science project can help with. To do so, Python can be used, and a model is created using TfidfVectorizer. Passive Aggressive Classifier can be implemented to distinguish between true and fake news. Pandas, NumPy, and sci-kit-learn are some Python packages suitable for this project, and we can utilize News.csv for the dataset.
# 

# In[1]:


#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


#theme setting
plt.style.use('ggplot')
sns.color_palette("tab10")
sns.set(context='notebook', style='darkgrid', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[20,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'


# In[8]:


#Read the data
df=pd.read_csv('fake_or_real_news.csv')
print(df.shape)
df.head()


# In[9]:


#Get the labels
labels=df.label
labels.head()


# In[10]:


target=df.label.value_counts()
target


# In[11]:


sns.countplot(df.label)
plt.title('the number of news fake/real');


# In[12]:


#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[13]:


#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[14]:


#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[15]:


#Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

