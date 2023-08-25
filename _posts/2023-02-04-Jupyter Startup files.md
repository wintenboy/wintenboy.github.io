---
layout: single
title:  "Jupyter Notebook Startup files"
categories: DL
tag: [Settings]
toc: true
use_math: true
typora-root-url: ../
sidebar:
  nav: "counts"
---

# Startup files setting 

+ Jupyter notebook 실행 시 미리 실행되는 코드

+ 설정 방법

  + Profile file 생성

    + ```bash
      $ ipython profile create
      ```

  + Startup file 수정

    + ```bash
      cd ~/.ipython/profile_default/startup
      ```

    + ```bash
      vi 00-first.py
      ```

+ ```python
  # basic
  import time
  import random
  
  # data analytics
  import numpy as np
  import pandas as pd
  
  # web crawling
  import requests
  from bs4 import BeautifulSoup
  
  # visulazation
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  sns.set()
  
  ```

  