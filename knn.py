import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm 

data =pd.read_csv(r"C:\Users\N\.spyder-py3\datasets\Dataset_spine.csv")

del data['Unnamed: 13']
#print(data.head())
data.isnull().any()

data.shape
"""
correlation = data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(correlation,annot=True)
plt.show()
"""
data[data['Class_att']=='Abnormal'].shape[0]
data[data['Class_att']=='Normal'].shape[0] 
"""
plt.figure(figsize=(15,10))
data.boxplot(patch_artist=True)
plt.show()
"""
data.drop(data[data['Col6']>400].index,inplace=True)

"""plt.figure(figsize=(15,10))
data.boxplot(patch_artist=True)
plt.show()
"""
data['Class_att']=data['Class_att'].apply(lambda x : '1' if x=='Abnormal' else '0') #abnormal 일경우= 1, nomarl 일경우= 2

data.reset_index(inplace=True) #인덱스 재배열
#print(data.head())

data_feature = data[data.columns.difference(['Class_att'])]  #데이터 스케일
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_feature)
data_scaled= pd.DataFrame(data=scaled_data,columns=data_feature.columns)
data_scaled['Class_att']=data['Class_att']
#print(data_scaled.head())

X=data_scaled[data_scaled.columns.difference(['Class_att'])]
Y=data_scaled['Class_att']

X=data_scaled[data_scaled.columns.difference(['Class_att'])]
y=data_scaled['Class_att']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)

Svm=svm.SVC()
Svm.fit(X_train,Y_train)
print("테스트결과:",100*Svm.score(X_test,Y_test))


