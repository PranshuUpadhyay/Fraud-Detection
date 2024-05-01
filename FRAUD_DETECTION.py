import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
def report(ytest,prediction):
 print(confusion_matrix(ytest, prediction))
 print(accuracy_score(ytest, prediction))
 print(precision_recall_fscore_support(ytest,prediction,average='binary'))

 
dataset = pd.read_csv('Variant V.csv')
print(dataset.head())
dataset.shape
print(dataset.columns)
print(dataset.dtypes)
print(dataset.isnull().sum())
dataset.dropna(inplace=True)
print(dataset.describe())

#visulizing dataset of fraud bool
fraud_inp = dataset['fraud_bool']
nonfraud_count=0
for i in fraud_inp:
 if i==0:
  nonfraud_count+=1
fraud_count = len(fraud_inp)-nonfraud_count


plt.bar(['Non-Fraud','Fraud'],[nonfraud_count,fraud_count], color=['red', 'green'])
plt.title('Fraud vs. Non-fraud Transcations')
plt.xlabel('Transcation Type')
plt.ylabel('Number of Transcations')
plt.show()



# Explore the distribution using a pie chart 
plt.figure(figsize=(6, 6))
dataset['fraud_bool'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
plt.title('Percentage of Fraudulent Transactions') 
plt.legend(['Not Fraud', 'Fraud'])
plt.show() 
print("Done plot")

#data preprocessing

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lst = ('payment_type','employment_status','housing_status','source','device_os')
for item in lst:
 dataset[item]=le.fit_transform(dataset[item])
print("Done encoding")

dataset=dataset.drop('device_fraud_count',axis=1) 
arr = list(dataset.columns)
lst = []
for col in arr:
 correlation = dataset['fraud_bool'].corr(dataset[col])
 print(f'Corellation of {col} is : {correlation}')
 lst.append(correlation)
print(dataset.columns)
print(lst)
print(sum(lst))
print(len(lst))
mean = sum(lst)/len(lst)
print(f'mean is :{mean}')
for cor in lst:
 if cor>=mean:
  
  print(cor)

x = dataset.iloc[ : , 1:].values
y = dataset.iloc[ : ,0].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x , y ,test_size=0.2,random_state=0)
print("Done test train split")

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print("done scaling")

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
y_pred1=classifier.predict(x_test)
print(y_pred1)
print("Logistic Regression Report: ")
report(y_test,y_pred1)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred2= classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred2)
print("Decision Tree Report: ")
report(y_test,y_pred2)



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)
y_pred3 = classifier.predict(x_test)
print("Random forest Report: ")
report(y_test,y_pred2)



from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train, y_train)
print("Model trained")
y_pred4 = classifier.predict(x_test)
print("SVM Report: ")
report(y_test,y_pred4)

