# Deepfake-Detection-Robust-on-adversarial-attack

## 과제 설계 배경 및 필요성
딥러닝 기술이 빠르게 발전함에 따라 점점 사람의 눈으로는 구별하기 어려운 딥페이크 영상이 생성되고 있다. 딥페이크 기술들은 잘못된 정보를 퍼트리거나, 개인을 괴롭히거나, 유명인의 명예를 실추시키는데 사용될 수 있다. 딥페이크가 악용되어 사회적인 피해를 줄 것을 대비하여 딥페이크 영상을 탐지하는 것에 대한 중요성이 커지고 있다 현재 딥페이크 영상을 탐지하는 다양한 기법들이 존재하며, 높은 정확도로 진짜 영상과 가짜 영상을 구분하고 있다. 다양한 탐지 기법 중에서도 CNN기반의 모델들이 좋은 성능을 보여주고 있기는 하지만, 이 모델들은 Adversarial Attack에 취약하다는 단점이 있다. 딥페이크 분야에서는 악의성을 가지고 충분히 학습된 탐지 모델을 속이는 적대적인 예제들을 생성해낼 가능성이 있기 때문에, 이러한 샘플들에 대해서 robust한 모델을 구축하는 것이 중요하다. 따라서 Adversarial attack에 robust한 딥페이크 탐지 모델을 구축하는 연구를 진행하였다.

## 과제 주요 내용
딥페이크 탐지 모델을 속일 적대적 예제를 생성할 attack 기법으로는 FGSM과 PGD를 사용하였다. 이때, 딥페이크 탐지에 사용되는 모델이 한정적이고 쉽게 이용 가능하게 공개되어 있기 때문에 whitebox setting을 가정하였다. 
딥페이크 탐지의 일반화 성능을 높여주는 Data Augmentation 방식인 Image Compression, Gaussian Blur, Gaussian noise를 적용한 이미지에 PGD 기법으로 적대적 예제를 생성하여 Adversarial Training 을 하고, 학습시 추가적으로 Lipschitz Regularization을 통해 gradient에 제약을 줌으로써 더욱더 robust한 딥페이크 탐지 모델을 구축하였다. 이 모델은 3가지 방식을 하나로 결합한 단일 프레임워크로, 원본 영상에 대한 정확도도 유지한 채, PGD와 FGSM으로 생성한 적대적 예제들을 90% 이상 defense한 강력한 모델이다. 이 방식은 데이터나 모델의 제약을 받지 않기 때문에 확장 가능하며, 기존에 강력한 defense 기법으로 알려져있던 DIP(Deep ImagePrior)에 비해 훨씬 짧은 시간 안에 높은 성능으로 동작 하므로 실생활에서도 충분히 활용 가능하다는 장점이 있다.

## 기본적인 Deepfake Detection Pipeline
프레임 별 CNN 기반의 분류 모델 
1. Dataset Download(video)
2. Dataset Proprocessing 
   : 각각의 비디오에서 원하는 개수의 프레임을 추출한다(전체 영상을 원하는 개수의 프레임만큼으로 구간을 나누어 추출하였다)
   이때, 모든 프레임에 대해서 Face Tracking 모델(retinaface)을 사용하여 인물의 face bounding box를 구한 후, 원하는 margin만큼 face crop하였다.
   
   <img width="414" alt="소캡디-중간발표 (1) - PowerPoint 2022-06-14 오후 6_02_15" src="https://user-images.githubusercontent.com/65711055/173538693-13afec34-afde-4de3-b11f-41adf5783aea.png">
3. Train model

   <img width="231" alt="소캡디-중간발표 (1) - PowerPoint 2022-06-14 오후 6_03_41" src="https://user-images.githubusercontent.com/65711055/173538938-e358be77-6913-492c-8f24-8ad7ffd8c1da.png">
4. Test model

## Defense Pipeline
![image](https://user-images.githubusercontent.com/65711055/173539463-e5c4fd47-2f2f-4386-b693-db4f223c47fe.png)
![image](https://user-images.githubusercontent.com/65711055/173539500-1f862af3-1e84-41fb-a20a-913db01e8b3b.png)
 
## 결과
### 1. baseline 모델
+ #### 파라미터 설정 <br>
```
  lr = 0.0001 <br>
  batchsize = 32 
  img_size = 299 
  face_margin = 0.3 
  epochs = 20 
  loss = BCEWithLogitsLoss 
  optimizer = Adam 
  scheduler = CosineAnnealingLR 
```
+ #### 결과
    Xception: Acc(97.683) / AUC(99.615)

### 2. baseline 모델에서의 Adversarial Attack success rate
#### Attack success rate : Perturbed Deepfake 영상 중 target label인 Real로 분류된 비율
![image](https://user-images.githubusercontent.com/65711055/173541104-5f17e86f-38b7-44d0-b90f-8846a1c0ed6c.png)
![image](https://user-images.githubusercontent.com/65711055/173541187-1d18f869-6c31-4e9b-b36e-6d40531e92a0.png)

### 3. Defense results
![image](https://user-images.githubusercontent.com/65711055/173539859-2539130b-d884-4483-961e-3457691239ef.png)

## 코드 설명
#### baseline model train code
``` python
python deepfake_detector/dfdetector.py 
--train True 
--dataset celebdf 
--data_path “your path”
--detection_method xception_celebdf 
--model_type xception 
--save_path “your path”
--epochs 20
```

#### desense model train code
``` python
python deepfake_detector/pgd_dfdetector.py 
--train True 
--dataset celebdf 
--data_path “your path”
--detection_method xception_celebdf 
--model_type xception 
--save_path “your path”
--augs strong
--epochs 50
```
