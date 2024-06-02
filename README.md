# AML_projectCode
2024-1 semester, advanced Machine Learning team projcet


## process explain

SVD이라는 데이터셋을 활용

    Saarbrücken Voice Database (SVD) : speech pathology detection dataset, 독일 음성, 1294개 훈련 데이터, 200개 테스트 데이터

적은 수의 데이터를 포함한 SVD 데이터를 가지고 높은 수준의 Accuarcy, F1 score를 얻어야함.


주어진 raw data는 .wav form file  =>  전처리 필요 (melspectrogram, MFCC, octave spectrogram)

"Voice pathology detection using convolutional neural networks with electroglottographic (EGG) and speech signals" 논문을 참고해 모델 시도

    시도한 모델 : renset50, CNN, GMM, randomForest, ensemble model, SVM

    사용한 전처리 기법 : melspectrogram, MFCC, octave spectrogram, Autocorrelation
    
    인지해야할 상황 : overfitting, convergence speed, available source, model size

가용 자원이 없어 단순히 colab에 의존해 모델 학습, 따라서 제한이 많은 상황.


## list up trying

1. resnet50(from scratch and pretrained) + melspectrogram
2. resnet50(from scratch and pretrained) + MFCC
3. resnet50(from scratch and pretrained) + ensemble(MFCC + melspectrogram)
4. CNN(pretrained) + octave spectrogram
5. GMM + melspectrogram
6. random forest + melspectrogram
7. SVM + Autocorrelation


## result

데이터가 적어서 이를 argumentation 하려 시도를 했지만 실패했다. colab 이라는 환경의 제약이 너무 컸다.

많은 시도 끝에 3번 모형에서 75% 근처의 accuracy를 얻었다. 기대에 못 미치는 정도이다. 여러가지 tuning을 해보며 기법을 추가해도 큰 발전이 보이지 않는다.

주어진 데이터와 딱 맞는 모델을 고르기가 쉽지 않다. Tuning 과정도 상당히 미숙한 모습을 보였다. 1~7번 모형에서 모두 vaildation set과 Test set간의 성능차이가 심했다.

코드 모듈화가 안되어있다. 이를 기능대로 분리하는 능력을 빨리 키워야한다.



