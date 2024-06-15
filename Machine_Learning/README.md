# Machine learing team project

## explain dataset

https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data
'Sleep Health and Lifestyle Dataset' 이라는 데이터 셋 사용
데이터 셋의 features는 다음과 같음


> 독립변수 : person ID, gender, Occupation, Age, sleep Duration, Quality of Sleep, Physical Activity level, BMI Category, Blood Pressure, Heart Rate, Daily step, Sleep Disorder
> 종속변수 : stress level


독립변수의 'Quality of Sleep'와 종속변수 'stress level' 은 서열형 변수형으로 1~10 단계로 되어있다.

'Blood Pressure'는 최고/최저 혈압의 형태로 표기되어 있고 이는 스트링 형에 담겨있다.

주어진 데이터 파일 csv는 train/test로 따로 존재하지 않고 오직 하나의 파일만 있다. 데이터 수는 374개이다.  매우 적음


우리의 목표 역시 주어진 데이터셋을 이용해 stress level을 예측하는 모델을 생성하는 것이다. 하지만 


## preprocessing


