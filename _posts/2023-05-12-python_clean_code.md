---
layout: single
title:  "Clean Code1"
categories: Python
tag: [basic]
toc: true
use_math: true
typora-root-url: ../
sidebar:
  nav: "counts"
---

+ 무심코 파이썬 코드를 작성해서 파이썬답지 못했던 코드들을 떠올리면서... 파이썬 클린코드라는 책을 읽고 재구성하였다.

# range(len()) 대신에 enumerate()

```python
items = ['book', 'key', 'pencil']
for i in range(len(items)):
  print(i, items[i])
```

위와 같이 리스트를 for 문에 넣고 무언가를 할 때, range(len())를 사용해서 index를 활용하곤 한다.

그러나 이와 같이 쓰는 코드는 규약은 굉장히 간단하지만 (실제로도 자주 보이곤 함) 이러한 방식보다 더 가독성이 뛰어난 코드는 다음과 같다.

```python
items = ['book', 'key', 'pencil']
for i in enumerate(items):
  print(i, items)
```

> 위의 두 코드를 모두 알고 있긴 했지만, 무심코 전자의 코드를 선호했던 것 같다. 앞으로는 좀 고치는 습관이 필요할 듯 하다. 

# 문자열 포매팅

+ r : 원시문자열 

파이썬에서는 백슬래시 \는 이스케이프 문자를 활용할 때 사용한다 (\n 이나 \t 등등). 특히 윈도우에서 파일 경로를 찾을 때 아래와 같이 주소를 쓰곤 한다.

```python
print('The file is in C:\\Users\\Al\\Desktop\\Info\\Archive\\Something')
```

이렇게 백슬래시를 두번 사용해서 파일 경로를 찾곤 하는데, 이러한 방식은 가독성이 떨어진다. 따라서 아래와 같은 코드가 더 파이썬답다고 할 수 있다.

```python
print(r'The file is in C:\Users\Al\Desktop\Info\Archive\Something')
```

r이 의미하는 바는 raw string (원시 문자열)을 의미한다. 이를 사용하면 백슬래시 문자를 이스케이프 문자로 받지 않고 문자열 리터럴을 사용하도록 해준다.

정규표현식을 사용할 때 사용하면 될 듯하다.

+ f - 문자열 (format-string이라는 뜻)

파이썬에서 print문을 활용할 때, f-문자열을 많이 사용한다. 이 방법에 대해서는 알고 있었지만 f-문자열은 파이썬 3.6부터 사용이 가능하다고 한다. 그리고 그 이전 버전은 format()함수 또는 %s 변환 지정자를 사용해야 한다. (사실 format함수와 %s 변환 지정자는 파이썬을 2년 넘게 써오면서 초창기 문법을 배울 때외엔 써본적이 없는 것같다.)

```python
name, age = 'mina', '22'
print(f'{name} is {age} years old')
print('{} is {} years old'.format(name, age))
print('%s is %s years old' % (name, age))
```

> 문자열 포매팅은 몰랐다기보다 어떤 약자를 의미하는지 알게 되었다.

# 파이썬 딕셔너리

+ get() 함수 

존재하지 않는 딕셔너리 접근 시 발생하는 에러 (KeyError)를 방지하고자 다음과 같은 코드를 사용하곤 한다.

```python
num_of_pets = {'rabbits':3}
if 'dogs' in num_of_pets:
  print(f'I have {num_of_pets['dogs']} dogs.')
else :
  print('I have 0 dogs')
```

그런데 이러한 것보다 좋은 것이 있으니... get()함수이다.

```python
num_of_pets = {'rabbits':3}
print(f'I have {num_of_pets.get('dogs', 0)}dogs')
```

위의 코드는 딕셔너리 내에 'dogs'라는 key가 존재하는지 찾고 있다면, dogs에 해당하는 values를 출력한다. 만약 없다면, 0을 출력한다.

+ setdefault() 함수

예를 들어 위와 같이 딕셔너리 내에 찾고자 하는 key가 존재하지 않는다면, 새로운 key를 지정하고 (이에 대응되는 value)를 할당하고 싶을 수 도 있다.

```python
num_of_pets = {'rabbits':3}
if 'dogs' not in num_of_pets:
  num_of_pets['dogs'] = 0
num_of_pets['dogs'] += 10
print(num_of_pets['dogs'])
```

이러한 흔한 경우에 대해서 는 setdefault()함수를 쓰면 좋다고 한다.

```python
num_of_pets = {'rabbits':3}
num_of_pets.setdefault('dogs', 0)
num_of_pets['dogs'] += 10
```

dogs라는 key를 찾고 있다면, 아무것도 하지 않는다. 그런데 dogs라는 key가 존재하지 않는다면 default로 0을 넣어서 key와 value를 만든다.

+ collections.defaultdict()

존재하지 않는 keys값에 계속해서 key와 value를 할당하고 싶을 수 있다. 그럴 때는 setdefault()함수보다는 collections.defaultdict()함수를 활용하는 것이 더 좋다. 마치 list에 계속해서 append하는 것처럼 사용할 수 있다 !

```python
>>> import collections
>>> ages = collections.defaultdict(int)
>>> ages
defaultdict(<class 'int'>, {})

>>> ages['mina'] += 22
defaultdict(<class 'int'>, {'mina': 22})
>>> ages['jason'] += 32
defaultdict(<class 'int'>, {'mina': 22, 'jason':32})
```

```python
>>> import collections
>>> room = collections.defaultdict(list)
>>> room['clean room'].append('vip')
>>> room['clean room'].append('vvip')
>>> len(room['clean room'])
2
>>> len(room['dirty room'])
0
```

+ switch문 to dictionary

```python
if season == 'Winter':
  holiday = 'New year\'s day'
elif season == 'Spring':
  holiday = 'May Day'
elif season == 'Summer':
  holiday = 'Juneteenth'
elif season == 'Fall':
  holiday = 'Halloween'
else :
  holiday = 'Personal day off'
```

위와 같이 if-elif-else문을 써서 장황하게 코드를 작성한 경험이 매우 많다. 항상 찝찝함을 느꼈지만 코드가 실행되는 것에만 집중하다 보니 이 코드를 깨끗하게 만들 생각은 해본 적이 없다. 아래와 같이 딕셔너리를 활용하면 위와 같은 코드를 깨끗하게 작성할 수 있다. 

```python
holiday = {'Winter': 'New Year\'s day',
           'Spring': 'May Day',
           'Summer': 'Juneteenth',
           'Fall': 'Halloween'}.get(season, 'Personal day off')
```

> 앞으로 이렇게 작성하는 것을 유용하게 써야겠다.

