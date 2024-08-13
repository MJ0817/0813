import matplotlib.pyplot as plt
# 회귀분석은 변수 간의 관계를 모델링휴ㅏ고 예측하는 통계적 기법
# 종류: 선형회귀, 비선형회귀

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv('./data/pima-indians-diabetes.data.csv',
                   names=header)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)


#데이터 분할
(X_train, X_test, Y_train, Y_test) = train_test_split(rescaled_X, Y, test_size=0.3)

#모델 선택 및 학습
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

#예측 정확도 확인
acc = accuracy_score(y_pred_binary, Y_test)
print(acc)

df_Y_test = pd.DataFrame(Y_test)
df_Y_pred_binary = pd.DataFrame(y_pred_binary)
# df_Y_test.to_csv("./results/y_test.csv")
# df_Y_pred_binary.to_csv("./results/y_pred.csv")
#결과(모델 예측값 VS 실제값) 시각화

plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', label='Predicted Values', marker='x')
plt.show()

plt.title("Comparison of Actual and Predicted values")
plt.xlabel("Index")
plt.ylabel('Class(0 or 1)')
plt.legend()
plt.show()
plt.savefig('./result/scatter.png')