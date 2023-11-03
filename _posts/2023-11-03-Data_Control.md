---
layout: single
title:  "큰 데이터를 메모리에 효율적으로 올리기"
categories: Python
tag: [preprocessing,coding]
toc: true
use_math: true
typora-root-url: ../
sidebar:
  nav: "counts"
---



# 큰 데이터를 메모리에 올리기 위한 효율적인 방법

## 준비동작

+ csv 형태의 데이터를 downcast한 뒤 parquet 데이터로 압축하여 데이터를 효율적으로 관리하기 위해서
+ pyarrow 와 fastparquet 패키지를 설치해야 함.


```python
!pip install pyarrow fastparquet
```

    Collecting pyarrow
      Downloading pyarrow-14.0.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (3.0 kB)
    Collecting fastparquet
      Downloading fastparquet-2023.10.1-cp310-cp310-macosx_11_0_arm64.whl.metadata (4.1 kB)
    Requirement already satisfied: numpy>=1.16.6 in /Users/winten/miniforge3/envs/data_control/lib/python3.10/site-packages (from pyarrow) (1.26.1)
    Requirement already satisfied: pandas>=1.5.0 in /Users/winten/miniforge3/envs/data_control/lib/python3.10/site-packages (from fastparquet) (2.1.2)
    Collecting cramjam>=2.3 (from fastparquet)
      Downloading cramjam-2.7.0-cp310-cp310-macosx_10_9_x86_64.macosx_11_0_arm64.macosx_10_9_universal2.whl.metadata (4.0 kB)
    Collecting fsspec (from fastparquet)
      Downloading fsspec-2023.10.0-py3-none-any.whl.metadata (6.8 kB)
    Requirement already satisfied: packaging in /Users/winten/miniforge3/envs/data_control/lib/python3.10/site-packages (from fastparquet) (23.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /Users/winten/miniforge3/envs/data_control/lib/python3.10/site-packages (from pandas>=1.5.0->fastparquet) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/winten/miniforge3/envs/data_control/lib/python3.10/site-packages (from pandas>=1.5.0->fastparquet) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in /Users/winten/miniforge3/envs/data_control/lib/python3.10/site-packages (from pandas>=1.5.0->fastparquet) (2023.3)
    Requirement already satisfied: six>=1.5 in /Users/winten/miniforge3/envs/data_control/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas>=1.5.0->fastparquet) (1.16.0)
    Downloading pyarrow-14.0.0-cp310-cp310-macosx_11_0_arm64.whl (24.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.0/24.0 MB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading fastparquet-2023.10.1-cp310-cp310-macosx_11_0_arm64.whl (682 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m682.6/682.6 kB[0m [31m4.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hDownloading cramjam-2.7.0-cp310-cp310-macosx_10_9_x86_64.macosx_11_0_arm64.macosx_10_9_universal2.whl (3.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.2/3.2 MB[0m [31m6.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading fsspec-2023.10.0-py3-none-any.whl (166 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m166.4/166.4 kB[0m [31m4.7 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: pyarrow, fsspec, cramjam, fastparquet
    Successfully installed cramjam-2.7.0 fastparquet-2023.10.1 fsspec-2023.10.0 pyarrow-14.0.0


![data_view](/images/2023-11-3-Data_Control/data_view.png)

# 대용량 데이터 불러오기

+ 대용량 데이터를 그냥 불러오게 되면 시간도 오래걸리고, 메모리도 비효율적임.


```python
import pandas as pd
%time df = pd.read_csv('data/국민건강보험공단_의약품처방정보_01_20211231.csv',encoding='cp949')
```

    CPU times: user 6.88 s, sys: 2.28 s, total: 9.16 s
    Wall time: 9.83 s



```python
df.shape
```




    (10500000, 15)



+ 실제로는 900MB이지만, pandas로 불러왔을 때는 1.2GB를 소모함.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10500000 entries, 0 to 10499999
    Data columns (total 15 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   STND_Y          int64  
     1   IDV_ID          int64  
     2   KEY_SEQ         int64  
     3   SEQ_NO          int64  
     4   SEX             int64  
     5   AGE_GROUP       int64  
     6   SIDO            int64  
     7   RECU_FR_DT      object 
     8   GNL_NM_CD       object 
     9   DD_MQTY_FREQ    float64
     10  DD_EXEC_FREQ    int64  
     11  MDCN_EXEC_FREQ  int64  
     12  UN_COST         float64
     13  AMT             int64  
     14  DATA_STD_DT     object 
    dtypes: float64(2), int64(10), object(3)
    memory usage: 1.2+ GB



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>STND_Y</th>
      <th>IDV_ID</th>
      <th>KEY_SEQ</th>
      <th>SEQ_NO</th>
      <th>SEX</th>
      <th>AGE_GROUP</th>
      <th>SIDO</th>
      <th>RECU_FR_DT</th>
      <th>GNL_NM_CD</th>
      <th>DD_MQTY_FREQ</th>
      <th>DD_EXEC_FREQ</th>
      <th>MDCN_EXEC_FREQ</th>
      <th>UN_COST</th>
      <th>AMT</th>
      <th>DATA_STD_DT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>628074</td>
      <td>9261</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>41</td>
      <td>2021-09-13</td>
      <td>347701ACH</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>521.0</td>
      <td>1563</td>
      <td>2022-08-11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>730013</td>
      <td>13348</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>41</td>
      <td>2021-02-17</td>
      <td>493801ATB</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>534.0</td>
      <td>1602</td>
      <td>2022-08-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021</td>
      <td>97734</td>
      <td>16827</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>41</td>
      <td>2021-06-10</td>
      <td>374602ATB</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>469.0</td>
      <td>1407</td>
      <td>2022-08-11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021</td>
      <td>818851</td>
      <td>20079</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>41</td>
      <td>2021-09-15</td>
      <td>367201ATB</td>
      <td>1.0</td>
      <td>1</td>
      <td>3</td>
      <td>764.0</td>
      <td>2292</td>
      <td>2022-08-11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>835362</td>
      <td>26258</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>41</td>
      <td>2021-07-17</td>
      <td>111501ATB</td>
      <td>1.0</td>
      <td>1</td>
      <td>30</td>
      <td>644.0</td>
      <td>19320</td>
      <td>2022-08-11</td>
    </tr>
  </tbody>
</table>

</div>



+ df.dtypes를 확인하게 되면, int"64" 그리고 float"64"와 같이 데이터 타입이 정해져 있음
+ 그러나 실제로 각 변수들이 int64를 쓰지 않아도 되는 변수일 수 도 있음.


```python
df.dtypes
```




    STND_Y              int64
    IDV_ID              int64
    KEY_SEQ             int64
    SEQ_NO              int64
    SEX                 int64
    AGE_GROUP           int64
    SIDO                int64
    RECU_FR_DT         object
    GNL_NM_CD          object
    DD_MQTY_FREQ      float64
    DD_EXEC_FREQ        int64
    MDCN_EXEC_FREQ      int64
    UN_COST           float64
    AMT                 int64
    DATA_STD_DT        object
    dtype: object




```python
import numpy as np
```


```python
df.select_dtypes(include=np.float64).max()
```




    DD_MQTY_FREQ      24240.0
    UN_COST         1450000.0
    dtype: float64



## downcast & to_parquet

+ csv 데이터를 chunk size만큼씩 불러와서 데이터가 가지고 있는 타입을 확인한 뒤 downcast를 해줌.
+ 적절한 타입(float, int, unsigned, string, category 등)으로 전부 바꿔주고
+ parquet data 형식으로 바꾸어 압축시켜줌.


```python
def downcast(df_chunk):
    for col in df_chunk.columns:
        dtypes_name = df_chunk[col].dtypes.name
        if dtypes_name.startswith('float'):
            df_chunk[col] = pd.to_numeric(df_chunk[col],downcast='float')
        elif dtypes_name.startswith('int'):
            if df_chunk[col].min() < 0:
                df_chunk[col] = pd.to_numeric(df_chunk[col], downcast='integer')
            else:
                df_chunk[col] = pd.to_numeric(df_chunk[col], downcast='unsigned')
        elif dtypes_name.startswith('object'):
            if df_chunk[col].__len__() > 20 :
                df_chunk[col] = df_chunk[col].astype('string')
            else:
                df_chunk[col] = df_chunk[col].astype('category')
    return df_chunk

chunksize = 1e6
chunks = []
file_name = '국민건강보험공단_의약품처방정보_01_20211231'
for chunk in pd.read_csv(f'data/{file_name}.csv',encoding='cp949', chunksize=chunksize):
    df_chunk = downcast(chunk)
    chunks.append(df_chunk)
    df_chunk.to_parquet(f'data_parquet/{file_name}_{df_chunk.index[0]}-{df_chunk.index[-1]}.parquet',index=False)
```


```python
from glob import glob
parquet_list = glob('data_parquet/*.parquet')
parquet_list
```




    ['data_parquet/국민건강보험공단_의약품처방정보_01_20211231_5000000-5999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_4000000-4999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_1000000-1999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_10000000-10499999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_0-999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_3000000-3999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_8000000-8999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_6000000-6999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_7000000-7999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_9000000-9999999.parquet',
     'data_parquet/국민건강보험공단_의약품처방정보_01_20211231_2000000-2999999.parquet']



# Parquet data 불러오기

+ csv로 불러왔을 때보다 훨씬 빠르고, 메모리도 효율적임.<img src="/images/2023-11-3-Data_Control/parquet_view.png" alt="parquet_view" style="zoom: 67%;" />


```python
%time df_parquet_list = [pd.read_parquet(gzip_file_name) for gzip_file_name in parquet_list]
len(df_parquet_list)
```

    CPU times: user 1.87 s, sys: 954 ms, total: 2.82 s
    Wall time: 2.32 s
    
    11


```python
%time df = pd.concat(df_parquet_list, ignore_index=True)
df.shape
```

    CPU times: user 394 ms, sys: 255 ms, total: 650 ms
    Wall time: 684 ms
    
    (10500000, 15)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10500000 entries, 0 to 10499999
    Data columns (total 15 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   STND_Y          uint16 
     1   IDV_ID          uint32 
     2   KEY_SEQ         uint32 
     3   SEQ_NO          uint8  
     4   SEX             uint8  
     5   AGE_GROUP       uint8  
     6   SIDO            uint8  
     7   RECU_FR_DT      string 
     8   GNL_NM_CD       string 
     9   DD_MQTY_FREQ    float32
     10  DD_EXEC_FREQ    uint8  
     11  MDCN_EXEC_FREQ  uint16 
     12  UN_COST         float32
     13  AMT             uint32 
     14  DATA_STD_DT     string 
    dtypes: float32(2), string(3), uint16(2), uint32(3), uint8(5)
    memory usage: 530.7 MB

