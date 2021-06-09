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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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
data['Class_att']=data['Class_att'].apply(lambda x : '1' if x=='Abnormal' else '0') #abnormal 일경우= 1, nomarl 일경우= 0

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
#수정x

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)


model = RandomForestRegressor(max_depth=10, min_samples_leaf= 32, min_samples_split=32, n_estimators=2000) #트리최대높이, 결정트리갯수,노드를 분할하기위한 최소한의 샘플, 리프노드가 되기 위한 최소한의 샘플데이터 
clf = model.fit(X_train,Y_train)

train_score = clf.score(X_train, Y_train)
test_score  = clf.score(X_test, Y_test)
print("테스트 결과:", 100*clf.score(X_test,Y_test))

"""
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, Y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(Y_test, pred)
print('랜덤 포레스트 정확도: {:.4f}'.format(accuracy))

model = RandomForestClassifier()



params = { 'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }


rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(X_train, Y_train)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확444도: {:.4f}'.format(grid_cv.best_score_))
"""