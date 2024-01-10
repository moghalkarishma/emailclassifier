#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv(r"C:\Users\mogha\Downloads\sms\spam.csv", encoding='ISO-8859-1')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


#1.data cleaning
#2.EDA
#3.Text Preprocessing
#4.Model Building
#5.Evaluation
#6.Improvement
#7.Website
#8.Deploy


# ## 1.Data cleaning

# In[6]:


df.info()


# In[7]:


#Drop last 3 columns 
df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)


# In[8]:


df.head()


# In[9]:


df.sample(5)


# In[10]:


df.rename(columns={"v1":"target","v2":"text"},inplace=True)


# In[11]:


df.sample(5)


# In[12]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df["target"]=l.fit_transform(df["target"])


# In[13]:


df.head(5)


# In[14]:


#Missing values
df.isnull().sum()


# In[15]:


df.duplicated().sum()


# In[16]:


df.shape


# In[17]:


df=df.drop_duplicates(keep="first")


# In[18]:


df.shape


# ## 2.EDA

# In[19]:


#Simply check how much percentage of sms are spam and ham
df["target"].value_counts()


# In[20]:


import matplotlib.pyplot as plt
plt.pie(df["target"].value_counts(),labels=["ham","spam"],autopct="%0.2f")
plt.show()


# In[21]:


# Data is Imbalanced 


# In[22]:


get_ipython().system('pip install nltk')


# In[23]:


import nltk


# In[24]:


#Count Number of characters every sms
#it prins Length every sms
df["num_characters"]=df["text"].apply(len)


# In[25]:


df.head()


# In[26]:


#no of Words
df["text"].apply(lambda x:nltk.word_tokenize(x))


# In[27]:


#no of Words
df["num_words"]=df["text"].apply(lambda x:len(nltk.word_tokenize(x)))


# In[28]:


df.head()


# In[29]:


#no of Words
df["text"].apply(lambda x:nltk.sent_tokenize(x))


# In[30]:


#no of Words
df["num_sentences"]=df["text"].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[31]:


df.head()


# In[32]:


df[["num_characters","num_words","num_sentences"]].describe()


# In[33]:


#ham here mean of number of characters used is less compared to spam messages
df[df["target"]==0][['num_characters',"num_words","num_sentences"]].describe()


# In[34]:


#spam here mean of number of characters used is more compared to ham messages
df[df["target"]==1][['num_characters',"num_words","num_sentences"]].describe()


# In[35]:


import seaborn as sns


# In[36]:


plt.figure(figsize=(12,6))
sns.histplot(df[df["target"]==0]["num_characters"],)#In ham-0
sns.histplot(df[df["target"]==1]["num_characters"],color='red')#In spam-1


# In[37]:


#Spam-1 made of more characters -red
#Ham-0 made of less characters -blue


# In[38]:


plt.figure(figsize=(12,6))
sns.histplot(df[df["target"]==0]["num_words"],)#In ham-0
sns.histplot(df[df["target"]==1]["num_words"],color='red')#In spam-1


# In[39]:


sns.pairplot(df,hue='target')


# In[40]:


df.corr()


# In[41]:


sns.heatmap(df.corr(),annot=True)


# In[42]:


#here target=1 means spam  here num_sentences ,num characters are 64% correlated and here num_sentences,num_words are 68% 
#correlated here num_characters and num_words are 97% correlated highly correlated
#here target is corrleated 38% with num_characters so we keep num_characters from 3 variables


# In[43]:


#Now I am to checking What top characters are used in spam and in ham


# ## 3.Data Preprocessing
# . Lower Case
# . Tokenization
# . removing special characters
# . Removing stop words and punctuation
# . Stemming

# In[44]:


#Stemming-means words like Loving,loves,loved are taken as love
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem("loving")


# In[45]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)


# In[46]:


from nltk.corpus import stopwords
stopwords.words("english")


# In[47]:


import string
string.punctuation


# In[48]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[49]:


df["text"][10]


# In[50]:


transform_text("I loved the YT lectures on Machine learning.How about you?")


# In[51]:


transform_text("Did you like my presentaion on ML")


# In[52]:


transform_text("Hi how are %% you Nitesh?")


# In[53]:


#Stemming-means words like Loving,loves,loved are taken as love


# In[54]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem("loving")


# In[55]:


df['transform_text']=df["text"].apply(transform_text)


# In[56]:


df.head()


# In[57]:


#Now here going to make Word Count for both spam and ham sms's


# In[58]:


get_ipython().system('pip install wordcloud')


# In[59]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=50,min_font_size=10,background_color='white')


# In[58]:


spam_corpus=[]
for msg in df[df['target']==1]['transform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[59]:


spam_corpus


# In[60]:


len(spam_corpus)


# In[61]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[62]:


ham_corpus=[]
for msg in df[df['target']==0]['transform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[63]:


len(ham_corpus)


# In[64]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# ## 4.Model Building

# In[67]:


#basically text classification we use Naive bayes basic algorithm
#here input and outputs must be in numerical here target is already numerical we need to make the transform text 
#must be converted into numerical
# To make text - into numerical is called vectorization
#methods-1.Bag of Words-making a column like on most frequent words  check count of each most frequent words
#2.tf-idf
#3.word to vector


# In[68]:


#This is based on bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[69]:


X=cv.fit_transform(df["transform_text"]).toarray()


# In[70]:


X


# In[71]:


X.shape


# In[68]:


y=df['target'].values


# In[69]:


y


# In[70]:


y.shape


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[77]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[78]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[79]:


gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[80]:


mnb.fit(x_train,y_train)
y_pred1=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[81]:


bnb.fit(x_train,y_train)
y_pred1=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[82]:


#here Imbalanced dataset We want precision to be maintain High from above bernoulli perform well


# In[126]:


#Now instead of bag of words I usetfidf Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()


# In[65]:


#with max_fetures =3000 in above without max_features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=3000)


# In[66]:


X1=tfidf.fit_transform(df["transform_text"]).toarray()


# In[143]:


#Scalling doesnot change either precision,accuracy instead it decrease both
#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler()
#X2=scaler.fit_transform(X2)


# In[ ]:


#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(X2,y,test_size=0.2,random_state=2)


# In[67]:


X1=np.hstack((X1,df['num_characters'].values.reshape(-1,1)))


# In[71]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=2)


# In[148]:


X1.shape


# In[72]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[73]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[74]:


gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[75]:


mnb.fit(x_train,y_train)
y_pred1=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[76]:


bnb.fit(x_train,y_train)
y_pred2=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[134]:


## tfidf-mnb
get_ipython().system('pip install xgboost')


# In[135]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[136]:


svc=SVC(kernel="sigmoid",gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver="liblinear",penalty='l1')
rfc=RandomForestClassifier(n_estimators=50,random_state=2)
abc=AdaBoostClassifier(n_estimators=50,random_state=2)
bc=BaggingClassifier(n_estimators=50,random_state=2)
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)
gbdt=GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb=XGBClassifier(n_estimators=50,random_state=2)


# In[137]:


clfs={
    'SVC':svc,
    'KN':knc,
    'NB':mnb,
    'DT':dtc,
    'LR':lrc,
    'RF':rfc,
    'AdaBoost':abc,
    'BgC':bc,
    'ETC':etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[138]:


def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[139]:


train_classifier(svc,x_train,y_train,x_test,y_test)


# In[140]:


accuracy_scores=[]
precision_scores=[]

for name,clf in clfs.items():
    
    current_accuracy,current_precision=train_classifier(clf,x_train,y_train,x_test,y_test)
    print("For",name)
    print("Accuracy - ",current_accuracy)
    print("precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[141]:


performance_df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores})


# In[142]:


performance_df


# In[ ]:


#with max_fetures =3000 in above without max_features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=3000)


# In[124]:


temp_df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores})


# In[125]:


performance_df.merge(temp_df,on='Algorithm')


# In[ ]:


new_df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores})


# In[ ]:


new_df_scaled=new_df.merge(temp_df,on='Algorithm')


# In[144]:


#sns.catplot(x='Algorithm',y='value',hue='variable',data=performance_df,kind='bar',height=5)

#plt.ylim(0.5,1.0)
#plt.xticks(rotation='vertical')
#plt.show()


# In[145]:


#model improve
#1.change the max_features parameter of tfidf as 300
#2.we done scalling
#3. appending the num_character col to x 


# In[149]:


#again do Accuracy,precision_scores calculation


# In[150]:


#voting classifier
svc=SVC(kernel='sigmoid',gamma=1.0,probability=True)
mnb=MultinomialNB()
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)

from sklearn.ensemble import VotingClassifier


# In[152]:


voting=VotingClassifier(estimators=[('svm',svc),('nb',mnb),('etc',etc)],voting='soft')


# In[153]:


voting.fit(x_train,y_train)


# In[157]:


y_pred=voting.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[158]:


#Apply Stacking
estimators=[('svm',svc),('nb',mnb),('etc',etc)]
final_estimator=RandomForestClassifier

from sklearn.ensemble import StackingClassifier


# In[161]:


clf=StackingClassifier(estimators=estimators,final_estimator=final_estimator)


# In[162]:



y_pred=voting.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[78]:


import pickle

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
            
            


# In[ ]:




