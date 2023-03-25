---
layout: single
title:  "[논문 리뷰] Handling Missing Data in Decision Trees: A Probabilistic Approach"
categories: ML
tag: [imputation, ML]
toc: true
use_math: true
typora-root-url: ../
---

**본 게시글은 개인적인 공부를 위해서 작성된 글이니 비약적인 내용 또는 틀린 내용이 포함되어 있을 수 있습니다**

# Abstract & Introduction

이 논문은 decision tree 기반의 모델을 활용하여 missing data를 다루는 방법에 방점을 두고 있다. 

decision tree는 machine learning model 들 중에서도 가장 각광받는 모델인데, 그 이유는 딥러닝이나 여타 다른 모델에 비해 갖는 장점이 있기 때문이다. 정리하자면,

+ Interpretability
+ ability to handle heterogeneous (mixed continuous-discrete) data
+ ability to handle missing data

결국 딥러닝과 같은 모델에 비해 해석가능하고, 각 columns들이 이질적인 형태를 보이는 데이터나 결측치가 있는 데이터들에 대해서 잘 작동한다는 것이다.

논문에서는 이러한 장점이 있는 decision tree를 활용하여 확률론적인 관점에서 missing data를 해결하는 방법을 제시하고 있다.

이 때, deployment time(배포)과 learning time(학습)에서 각각 확률론적인 방법을 사용해서 missing data를 처리하는 것을 Main Contribution으로 보여주고 있다.

# Background

+ notation 
  + $X$ : RVs 확률변수
  + $x$ : $X$로부터의 관측값
  + $X^o$ : 관측변수
  + $X^m$ : 결측치가 포함된 변수
+ Decision Tree
  + 생략
+ Decision Forests
  + 생략
+ Decision trees for missing data
  + mean, median, mode imputation & PVI(predictive value imputation) : model의 Input으로 들어가기 전에 missing data가 처리되는 방식
    + 데이터 분포에 대한 강력한 가정이 필요함. (때로는 적절하지 못할 수 도 있음)
  + Mulitple imputation with chained equations, surrogate splits(used in CART), MIA(XGBoost) : decision tree에서  

# Method

## 1. Expected Predictions of Decision Trees

## 2. Expected Parmeter Learning of Trees

# Experiments

# Conclusion

