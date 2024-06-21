# Probabilistic machine learning team project

통계적 기계학습 팀 프로젝트는 다른 머신러닝 프로젝트와 다르게 주어진 데이터셋에 대한 EDA 부터 시작하여 통계적인 근거에 따라 모델을 구축한다.

## dataset

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





## model implement





## result and analysis





