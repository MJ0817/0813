#Decision tree 의사결정나무

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv('./data/pima-indians-diabetes.data.csv',
                   names=header)
#데이터 전처리: Min-Max 스케일링
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)

#데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y, test_size=0.3)

#max_depth= =노드의 개수(트리의 깊이)
model = DecisionTreeClassifier(max_depth=1000, min_samples_split=50, min_samples_leaf=5)
#학습
model.fit(X_train, Y_train)

#예측값 생성
y_pred = model.predict(X_test)
print(y_pred)

#모델 정확도
acc = accuracy_score(Y_test, y_pred)
print(acc)