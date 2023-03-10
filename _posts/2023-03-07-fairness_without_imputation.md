---
layout: single
title:  "[논문 리뷰] Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values"
categories: machine learning
tag: [imputation,machine learning, missing values]
toc: true
---

# Abstract & Introduction

## Abstract

본 논문에서는 missing values가 있는 데이터셋에 대해서 머신러닝 모델을 적용할 때, 발생하는 Fairness concerns에 대해서 다루고 있다. 공정성에 개입하여 학습시키는 모델들이 여러 가지 제안되어 왔는데(MIP 모델 등 뒤에서 설명) , 이러한 모델들은 보통 완전한 학습 datasets을 가정하고 있다. 따라서 missing values가 있는 불완전한 datasets들에 대해 imputation(결측값 대치)를 한 뒤 공정성에 개입하는 머신러닝 방법들이 적용된다면, 불공정한 모델이 만들어질 수도 있다. 그 이유는  

***

여기서 Fairness라는 표현이 모호한데, 공정성이라고 해석하지만 모델이 얼마만큼 편향되었는지를 이야기하는 것이라고 생각하면 좋다. 또, 모델이 편향되었다는 의미는 대표적인 한 기업의 사례로 설명할 수 있다.

+ **Example case** : Northpointe사가 만든 재범 확률을 예측하는 알고리즘 COMPAS가 이를 이해하기 쉬운 예시이다. COMPAS 알고리즘은 피고인의 일반범죄와 강력범죄 각각에 대한 재범률을 측정하여 피고인에 대한 risk score(1~10)를 판사에게 제공한다. 그리고 판사는 이를 보조자료의 형태로써 활용하여 형량을 선고 및 가석방에 대한 도움을 받을 수 있다. 그런데 2016년 이 알고리즘이 인종편향적인 결과를 보여준다는 통계가 나오게 된다. 백인의 경우 risk score가 1에 가장 많이 분포하고 10까지의 비율이 점차 감소하는 반면, 흑인의 경우 risk score가 1~10까지 고루 분포한다. '여기까지만 들었을 때는 그럴 수도 있지 않겠느냐'라는 생각이지만 조금 더 설명을 이어 가보겠다. 당시 재범률이 높은 것으로 예측되었지만 실제로 2년간 범죄를 저지르지 않은 것으로 드러난 경우가 흑인은 45%, 백인은 23%로 두배에 달하였다. 그리고 재범률이 낮은 것으로 예측되었지만 실제로 2년간 범죄를 저지른 경우가 백인이 48%, 흑인이 28%였다. 각각, 약 두배에 달하는 차이이고, 이는 모델이 정확성을 떠나서 공정하지 못하고 인종편향적이라고 이야기 할 수 있다.

***

그래서 논문에서는 결측값이 포함된 상태에서 학습이 가능한 decision tree 기반의 모델을 제시하고 있다. abtract이기 때문에 간단하게만 설명하면, 결측치가 포함되어 있는 속성으로 decision tree를 하되 fairness-regularized objective function으로 최적화 과정을 수행한다. 

마지막으로 논문에서는 제시하는 방법을 통해 imputed dataset에 fairness intervention method (model)를 적용하는 것보다 더 좋은 성능을 보이고 있다.

## Introduction

abstract에서 이야기하였지만, 논문이 가장 중요하게 쳐다보고 있는 곳은 "missing values"와 "fairness"이다. missing values는 모델 학습과 관련해서 유의미한 영향을 미치는 경우가 있을 수 있다. 이렇게 missing values가 발생하는 경우는 여러가지가 있다. 랜덤하게 발생하는 경우가 대부분이지만 인간과 관련된 데이터의 경우 사회인구학적인 특성(인종, 소득, 나이)과 연관되어 발생하는 경우도 있다. 예를 들면, 저소득 환자가 비용이 많이 드는 의료검사를 거부해서 결측값이 생기는 경우가 있다. 또, 글씨 pont의 크기가 너무 작아서 나이가 많은 사람들이 설문에 응답하지 못했거나, Non-native speakers에게 너무 어려운 언어로 설문이 되어 있어서 발생하는 결측값이 있을 수 도 있다. 

> 이러한 경우, 상식적으로 어떤 값을 임의로 대치해서 모델 학습 (training model) 또는 데이터 분석(data analysis)을 수행하는 것은 적절하지 않아보인다 !

머신러닝의 기본적인 pipline을 생각해보자. 데이터가 모델로 들어가기 전에 먼저 전처리 단계를 거치게 된다. 그리고 전처리 단계 내에는 우리가 다루고자 하는 missing values들을 다른 값으로 대치(performing imputation)하거나 삭제(dropping missing values) 하는 단계가 있다. 하지만, 앞서 이야기했듯이, 이러한 전처리 방식은 모델의 성능을 떠나서 편향된(biased) 모델, 즉 공정하지 못한 모델이 만들어질 수 있다. 본질적으로, 이렇게 편향된 결과가 나올 수 있는 우려가 있는 데이터를 가지고 앞서 잠깐 언급한 Fairness한 머신러닝 모델들을 적용한다면, 어떻게 될까? 

논문에서는 이러한 문제에 해결하기 위해서 missing values 처리와 fairness한 머신러닝 모델들을 연결하여 해결책을 제시하고 있다. 이 때, missing values의 처리 방식이 fairness에 어떠한 영향을 미치는지 이론적으로 분석하고, real-world datasets들을 활용해서 제시하고 있다.

missing values를 임의의 값으로 대치함으로써 발생하는 잠재적인 discrimination risk에 대한 이론적인 분석은 다음과 같은 세 가지 요인을 중점으로 살펴보게 된다.

1. imputation 방법의 성능이 각 group attributes마다 다르게 나타날 수 있다. (group attributes란 한 개인을 어떤 그룹으로 특정할 수 있게 해주는 속성들을 이야기한다. tabular data에서 봤을 때, 인종이나 성별, 연령 같은 columns의 attributes라고 볼 수 있다.) 그 결과 imputated data는 모델의 bias에 영향(Inherit and propogate)을 미치게 된다. 
2. 
