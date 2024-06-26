import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import GridSearchCV
import sklearn.svm as svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


###########################   data preprocessing     ###########################

# CSV 파일을 읽기, dtype : DataFrame
stress = pd.read_csv('drive/MyDrive/Sleep_health_and_lifestyle_dataset.csv')
stress = stress.drop(['Person ID', 'Occupation'], axis=1)


# BMI category 를 원핫인코딩, 정수형으로 메핑
weight_categories = {'Normal': 0, 'Normal Weight': 0, 'Obese': 1, 'Overweight': 2}
stress['BMI Category'] = stress['BMI Category'].map(weight_categories)


# 혈압 데이터를 두 개의 특성으로 나누기
stress[['High Blood Pressure', 'Low Blood Pressure']] = stress['Blood Pressure'].str.split('/', expand=True)

# 데이터 타입을 정수형으로 변환
stress['High Blood Pressure'] = stress['High Blood Pressure'].astype(int)
stress['Low Blood Pressure'] = stress['Low Blood Pressure'].astype(int)

# 분할된 혈압 데이터를 포함한 새로운 데이터프레임 생성
stress = stress.drop('Blood Pressure', axis=1)


# 수면장애가 없는 데이터로 필터링

print(stress.shape)
stress_fittered = stress[~stress['Sleep Disorder'].isin(['Sleep Apnea', 'Insomnia'])]

# 이제 수면장애가 없는 사람들만 모였으므로 해당 칼럼 삭제
stress_fittered = stress_fittered.drop('Sleep Disorder', axis=1)

print(stress_fittered.shape)

stress_fittered.tail()

# feature number : 13 > 10(delete ID,Occupation,Disorder) > 11(split blood Pressure)

# convert csv file to tensor

continuous_data = stress_fittered.select_dtypes(include=['int', 'float'])
categorical_data = stress_fittered.select_dtypes(exclude=['int', 'float'])



categorical_data_encoded = pd.DataFrame()

for column in categorical_data.columns:
    unique_categories = categorical_data[column].unique()
    category_mapping = {category: i for i, category in enumerate(unique_categories)}
    print(category_mapping)
    categorical_data_encoded[column] = categorical_data[column].map(category_mapping)

stress_tensor = pd.concat([continuous_data, categorical_data_encoded], axis=1).to_numpy()


total_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category'
              , 'Heart Rate','Daily Steps', 'High Blood Pressure', 'Low Blood Pressure', 'Gender']

train_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'BMI Category'
              , 'Heart Rate','Daily Steps', 'High Blood Pressure', 'Low Blood Pressure', 'Gender']


y_train = stress_tensor[:, 4]
X_train = np.concatenate((stress_tensor[:, :4], stress_tensor[:, 5:11]), axis=1)





# 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# 라벨값을 클래스로 변환하는 함수 , 2진분류로 변환
def value_to_label(label):
    if label >= 3.0 and label < 5.0:
        return 0
    elif label >= 5.0 and label < 7.0:
        return 1
    else :
        return 2


# 라벨값을 클래스로 변환
y_train = np.array([value_to_label(label) for label in y_train])


X_train_scaled.shape, X_train_scaled , y_train, X_train_scaled[0]


###########################   feature analysis     ###########################


from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import GridSearchCV

# 탐색할 alpha 값 범위 설정, 약한 규제
alphas = np.arange(0.01, 0.11, 0.01)

best_alpha = None
best_model = None
best_remaining_features = None

for alpha in alphas:

    lasso = Lasso(alpha=alpha)

    # lasso로 grid search, cv : cross vaildation
    model = GridSearchCV(estimator=lasso, param_grid={'alpha': [alpha]},
                         cv=10, scoring='neg_mean_squared_error')
    model.fit(X_train_scaled, y_train)

    remaining_features = [train_cols[i] for i in np.where(model.best_estimator_.coef_ != 0)[0]]
    print(f"Alpha: {alpha}, Remaining Features: {remaining_features}, Model score: {model.best_score_}")

    if best_model is None or model.best_score_ > best_model.best_score_:
        best_model = model
        best_alpha = alpha
        best_remaining_features = remaining_features

# 최적의 모델 및 결과 출력
print(f"\nBest Alpha: {best_alpha}")
print(f"Best Model: {best_model}")
print(f"Remaining Features with Best Model: {best_remaining_features}")




X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

y_train_tensor = y_train_tensor.view(-1, 1)

data_tensor = torch.cat((X_train_tensor, y_train_tensor), dim=1)

correl_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'BMI Category'
              , 'Heart Rate','Daily Steps', 'High Blood Pressure', 'Low Blood Pressure', 'Gender', 'Stress level']




# 상관 행렬 계산
correlation_matrix = np.corrcoef(data_tensor.numpy(), rowvar=False)

# 상관 행렬을 히트맵으로 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            xticklabels=correl_cols,
            yticklabels=correl_cols)
plt.title('Correlation Matrix')
plt.xlabel('Features and Label')
plt.ylabel('Features and Label')
plt.show()


X_train_lasso = X_train_scaled[:,[0,1,2,3,4,5,6,7,9]]


###########################   model training     ###########################
#  sigmoid kernel + 9 features

svm_clf =SVC(kernel = 'sigmoid')

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'coef0': [0, 0.1, 0.5, 1, 5, 10]
}


# GridSearchCV를 사용하여 하이퍼파라미터 탐색
grid_search = GridSearchCV(svm_clf, param_grid, cv=10, scoring='accuracy', verbose=1)
grid_search.fit(X_train_lasso, y_train)

# 최적의 하이퍼파라미터 출력
print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("테스트 세트 정확도:", grid_search.best_score_)
