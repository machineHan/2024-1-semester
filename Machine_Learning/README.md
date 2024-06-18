# Machine learing team project

## explain dataset

    https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data
    
'Sleep Health and Lifestyle Dataset' 이라는 데이터 셋 사용

데이터 셋의 features는 다음과 같다.


> 독립변수 : person ID, gender, Occupation, Age, sleep Duration, Quality of Sleep, Physical Activity level, BMI Category, Blood Pressure, Heart Rate, Daily step, Sleep Disorder
> 종속변수 : stress level


독립변수의 'Quality of Sleep'와 종속변수 'stress level' 은 서열형 변수형으로 1~10 단계로 되어있다.

'Blood Pressure'는 최고/최저 혈압의 형태로 표기되어 있고 이는 문자형에 담겨있다.

주어진 데이터 파일 csv는 train/test로 따로 존재하지 않고 오직 하나의 파일만 있다. 데이터 수는 374개이다 => 매우 적음


우리의 목표 역시 주어진 데이터셋을 이용해 stress level을 예측하는 모델을 생성하는 것이다. 다만 1~10 range의 종속변수를 예측하지 않고 스트레스를 범주형으로 나눠 스트레스 높음/중간/낮음 으로 분류하는 모델을 만들 것이다.


## preprocessing

프로젝트에서 원하는 기능은 직관적으로 측정가능한 수치를 가지고 '일반적인' 상황의 유저에게 스트레스 정도를 예측해서 알려주는 어플리케이션을 개발하고 싶었다. 이런 이유하에 Sleep Disorder 항목에 'None'이라는 항목만을 추출하여 사용했다. 수면장애를 가진 항목을 제외하니 300개 미만의 데이터가 뽑혔다.

주관적인 견해로 person ID, Occupation 은 종속변수 'stress level'에 큰 영향을 주지 않을 것이라 생각하고 이를 제외하여 모델은 구축했다. 우리가 사용할 데이터는 300개 미만으로 직업에 대해 구분해서 예측하기에는 데이터가 부족하다는 견해로 인해 직업항목도 제외하였다.

BMI Category는 총 4개의 항목으로 구성되어있다.

> BMI Category : 'Normal', 'Normal Weight', 'Obese', 'Overweight'

이에 대해 정확한 설명이 없어 이를 각각 0,0,1,2 로 Onehot Encoding하여 사용했다.

Blood Pressure는 '최고혈압/최저혈압' 형태의 문자열로 저장되어 있다. 이를 / 기준으로 앞뒤로 분리하고 각각을 'High Blood Pressure', 'Low Blood Pressure'로 나누어 열에 추가했다. 문자열 1개의 특징에서 정수형 2개의 특징으로 분할되었다.


## method


데이터를 상관관계분석을 해서 비선형관계에 있다고 보았다. 이를 통해 선형기반의 ML 모델은 배제했다. 후보 모델로 kernel SVM, 트리 기반 모델을 생각했다.

사용 중인 데이터를 보면 대부분이 연속형 자료형이다. 트리 기반의 모델은 연속형 변수에 대한 오분류가 크다는 점을 인지해 트리 기반의 모델보다 kernel SVM을 우선시했다.

성능이 좋은 커널을 실험적으로 찾는 방식을 택해서 gridSearch를 통해 가장 높은 정확도를 갖는 커널을 사용했다.

실험을 진행하며 문득 다음과 같은 생각이 들었다

> 주어진 모든 데이터를 다 사용하는 것이 항상 좋을까?

그리하여 이를 비교하기 위해, 전체 feature를 사용해 훈련한 모델과 lasso regression을 통해 골라진 feature sebset를 이용해 훈련한 모델 둘을 비교하기로 했다.

결과는 lasso regression의 Alpha: 0.01 에서 선택된 features를 가지고 (low blood pressure만이 제외됨) 훈련한 sigmoid kernel SVM이 가장 높은 성능을 보였다.


## result and analysis

모든 특성을 사용한 경우보다 ‘저혈압(low blood pressure)’ 특성을 제외한 경우가 더 좋은 성능을 나타냈습니다. 모든 데이터를 사용하여 훈련하는 것이 좋다고 생각되지만, 이를 맹목적으로 따르는 것은 좋지 않다는 것을 알 수 있었다.

현재 데이터는 약 300개로 실험한 모델의 설정 값을 사용해도 문제가 없지만, 데이터 양이 증가하면 계산 비용이 크 게 증가할 수 있습니다. 이러한 상황에선 lasso regression의 규제 정도를 강화하여 더 적은 특성으로 학습시키면 계산 비용을 줄일 수 있을 것이다.

결론적으로, 특성 선택에 있어서는 lasso와 같은 규제 기법이 유용하며, 직관적 상식에만 의존하는 것은 한계가 있음을 확인.

또한, 데이터의 양과 특성의 선택은 모델의 성능과 비용에 큰 영향을 미친다는 점을 알 수 있었다.

