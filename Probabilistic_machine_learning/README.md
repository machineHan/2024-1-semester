# Probabilistic Machine Learning Team Project

통계적 기계학습 팀 프로젝트는 다른 머신러닝 프로젝트와 다르게 주어진 데이터셋에 대한 EDA 부터 시작하여 통계적인 근거에 따라 모델을 구축한다.

## Dataset

> https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data


`Airline Passenger Satisfaction`이라는 항공사 데이터 셋을 사용했다. 캐글에서 제공된 데이터셋의 대한 설명을 보고 '추측한' 바에 의하면 일정기간 내의 한 항공사에서 실시한 설문조사 데이터를 가지고 만든 데이터셋으로 이해할 수 있다.

데이터셋은 다음과 같이 이루어져있다.

> 독립변수 : id, Gender, Customer Type, Age, Type of Travel, Class, Flight Distance, Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Inflight service, Cleanliness, Departure Delay in Minutes, Arrival Delay in Minutes

> 종속변수 : satisfaction


보는 바와 같이 독립변수가 매우 많다. 주어진 데이터의 갯수는 train/test dataset가 각각 103904/25976 개로 학습하기에 충분한 데이터셋을 포함하고 있다.

id 변수는 예측에 필요 없으므로 삭제한다.

이 데이터셋은 `설문조사`이다. 대부분의 feature들이 서열형 변수로 표기되어 있다. 서비스의 정도를 나타내는 변수들은 대부분 0~5의 점수를 포함하고 있다.

하지만 점수 0이 뜻하는 바가 1보다 작은 서비스에 대한 점수인지에 대한 설명이 케글의 데이터 제공자에 의해 정확히 정의되어 있지 않았다. inflight wifi service에 대한 0은 `Not Applicable`이라고 정의돼 있지만 나머지는 정의없이 0으로 표기되었다. 이는 추후 EDA를 통해 사용여부를 결정했다.

범주형 변수는 다음과 같이 포함되어있다. 

    Gender : Female, Male
    Customer Type : Loyal Customer, Disloyal Customer
    Type of Travel : Business travel, Personal travel
    Class : Business, Eco, Eco Plus
    satisfaction : satisfied, neutral or dissatisfied






## EDA and preprocessing

훈련 데이터와 테스트 데이터가 동일한 분포에서 왔다는 가정해에 EDA를 훈련 데이터에서만 실행하고 테스트에서 동일하게 적용하겠다.


훈련 데이터에 대해 Arrival Delay in Minutes에 결측치 310개가 존재한다. 이는 훈련 데이터 0.3%로 이를 제거해도 유의미한 편향이 생길 것으로 보기 힘들다. 해당 행을 삭제했다.

반복해서 말하지만 만족도 데이터 셋은 `설문조사`이다. 즉 노이즈가 많을 수 밖에 없는 상황이다. 다음은 설문조사 데이터에 대해 다뤄야할 부분 중 일부이다.

    - 성실하게 답변한 설문지와 그렇지 않은 설문지
    - 각 서비스 평가 항목의 간극에 대한 논의 ex) 1>2 사이의 1점과 4~5 사이의 1점은 동일한 간격인가
    - 비슷한 서비스에 대해 점수가 상극 ex) Seat comfort and Seat comfort
        

노이즈를 피하기 위해 도메인 지식으로 최대한 전처리가 필요한 상황이다. 하지만 노이즈를 제거하는 것은 매우 어렵다. 그리하여 일반적인 상식에서 통할 수 있는 도메인 지식을 통해 간단한 전처리를 하게 되었다.

    1. 모든 항목이 같은 설문은 제거
    2. 서비스 항목에서 0이 한 번이라도 포함된 데이터 제거

1,2번의 이유는 일반적인 통념하에 생각한 방식이다. 

1번이 가능한 이유는 다음과 같다. 설문조사는 객관적인 지표라고 하기에는 어렵다. 고객의 기분에 따라 매우 유동적으로 변한다. 그리고 같은 비행기를 타도 그에 대한 설문조사의 답은 천차만별이다. 그날 좋은 일이 있었던 고객은 모든 항목을 5점을 주었을 수 있고, 특정 서비스에서만 불쾌감을 느꼈지만 모든 항목에 1에 가까운 점수를 줄 수도 있다. 이는 객관적인 지표나 자료에 따른 지식이라기 보단 경험적인 접근이다. 그리하여 모든 항목에 같은 점수를 준 설문지는 성의있게 작성한 설문지라고 보기 어렵다는 결론에 도달하여 이를 삭제하여 사용하겠다.


2번의 이유는 데이터 제공자의 설명부족이다. inflight wifi service의 0점은 설명이 가능하여 어떻게 다룰지가 명백하다. 하지만 나머지 0점은 1점보다 적은 점수로써 보는 것이 맞는지 아지면 적용불가능한 점수로 보는 것이 옳은지에 대한 판단이 쉽지 않다. 그리하여 이를 모두 제거하여 삭제해본다는 결론에 도달했다. 

1,2번에 해당한 데이터를 삭제했고 훈련 데이터가 103904 -> 95682 로 줄어들었다. 하지만 여전히 많은 데이터 포인트가 존재해서 이를 문제없이 다룰 수 있다.


범주형 feature들을 Onehot Encoding 방식을 통해 정수형 변수로 변경하였다. 위에서 범주형 변수를 얘기한 순서대로 각각 0부터 오름차순으로 정수를 부여했다.




## model implementation

EDA를 통해 데이터 전처리를 거치면 모든 feature들이 정수형 변수로 변환된다. 

사용할 ML algorithm을 리스트업하면 다음과 같디

1. logistic regression
2. kernel SVM
3. Random Forest
4. KNN-PCA


데이터-모델 적합성, 모델 예측도, 모델 설명력 이런 순서를 통해 모델을 선정했다.

logistic regression은 데이터를 선형적으로 분리하는 회귀식이다. 하지만 모든 독립변수, 종속변수간의 correlation을 확인한 결과 선형 관계가 불충분하다고 판단되어 이를 제외했다.


SVM은 클래스를 가장 효과적으로 나누는 선형 초평면을 찾는 방식의 모델로 만약 데이터가 선형 분리가 불가능하다면 고차원의 초평면(현재 차원의 비선형 초평면)을 찾을 수 있다. kernel로 인해 설명력이 다소 떨어지는 결과를 가진다 해도 분류문제를 해결할 수 있는 강력한 모델이다. 이런 인사이트를 통해 kernel SVM은 설명력이 다소 아쉽지만 데이터-모델 적합도가 높고, 예측력 또한 높을 것이라 예상되었다.


Random Forest는 다수의 decision tree를 불안정하게 성장시켜 많은 모델 다양성을 확보한 뒤 이를 앙상블하여 사용하는 ML algorithm이다. 앙상블 특유의 일반화 성능이 높다는 점과 일반적으로 다양한 상황에서 잘 작동한다는 점에 기인해 이를 사용하면 좋은 결과를 얻을 수 있을 것이라 생각했다. 트리 기반모델이기에 모델-데이터 적합도는 말할 것도 없다.


KNN은 인접한 k개의 데이터들을 통해 해당 데이터가 어떤 클래스인지 확인하는 ML algorithm이다. 하지만 고차원 공간에서 좋은 성능을 내지못한 다는 점이 있다. 이를 개선하기위해 특성 축소방법중 하나인 PCA를 적용하면 좋은 결과를 낼 수 있을 것이라 예상했다. kernel SVM과 마찬가지로 PCA로 인한 설명력 감소가 있지만, 모델-데이터 적합도, 예측력이 높을 것이로 에상되어 이를 시도했다.



<br><br>

결론부터 말하면 KNN-PCA, kernel SVM은 너무 많은 자원이 필요하며 예측성능이 그에 비해 많이 아쉬웠다. 그에 반해 random forest는 작은 코스트로 높은 성능을 보여주었다.

gussian kernel을 통한 training은 100분이 소모되었고 검증셋에 대해 92% 근방의 성능을 보였다.

KNN-PCA 역시 마찬가지 였다. 높은 성능을 보일려면 기본 독립변수와 비슷한 20개 근방의 주성분을 사용해야하지만 이 상황에선 2시간이 넘는 훈련시간이 걸려 이를 폐기했다.

Random Forest는 학습시간이 2분 내외로 걸리고 여러가지 하이퍼파리미터가 존재해 모델을 더 좋게 꾸밀 수 있다. GridSearchCV를 통해 가장 적절한 하이퍼파라미터를 찾았고 성능은 96% 이상이 나왔다.




## result

Random Forest가 가장 좋은 모델로 선택되었다. 데이터-모델 적합도, 예측도가 높아서 모델 고려사항에 잘 맞았다. 개별 트리가 수백개여서 모델 설명력은 낮아지지만 이를 보조하는 feature importance tool이 존재해 어느정도 커버할 수 있었다. 즉 모델 고려 사항 3가지에 모두 적합한 모델이다.

성능은 다음과 같이 나왔다

    accuracy : 96.13%
    f1 score : 95%


## additional EDA

feature importance를 통해 top5의 특성 중요도를 보면 다음과 같다.

1. Online boarding
2. Inflight wifi service
3. Class
4. Type of Travel
5. Inflight entertainment

이를 위주로 개선을 한다면 다음 설문에 더 좋은 만족도를 기대할 수 있을 것이다.

<br><br>


3,4번의 특성 중요도는 범주형 독립변수이다. 여기서 각 범주 사이의 차이가 극명하게 갈리는 상황을 포착했다.

예를 들어 Type of Travel에 대해 만족 만족/불만족이 크게 엇갈리는 결과를 보였다. 업무 목적으로 가는 승객은 58.3%의 승객이 만족을 표시하는데 반해 개인적인 용무로 비행기를 이용하는 승객은 만족 비율이 10.2%로 굉장히 낮은 만족도를 보였다.

Class에서도 이런 양상이 보여진다. Business Class 승객은 69.4%가 만족이라고 표기한 반면, Eco Class 승객은 19.4%만 만족했다.

이런 인사이트를 통해 고객을 세분화하여 맞춤 전략을 짜서 고객별 서비스 향상에 힘쓰는 것이 바람직하다고 보여진다.






