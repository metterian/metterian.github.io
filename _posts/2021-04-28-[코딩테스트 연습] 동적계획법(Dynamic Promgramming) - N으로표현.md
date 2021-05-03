---
layout: post
title: "[코딩테스트 연습] 동적계획법(Dynamic Promgramming) - N으로표현"
author: "metterian"
tags: 프로그래머스 알고리즘
---
# [코딩테스트 연습] 동적계획법(Dynamic Promgramming) - N으로표현

###### 문제 설명

아래와 같이 5와 사칙연산만으로 12를 표현할 수 있습니다.

12 = 5 + 5 + (5 / 5) + (5 / 5)
12 = 55 / 5 + 5 / 5
12 = (55 + 5) / 5

5를 사용한 횟수는 각각 6,5,4 입니다. 그리고 이중 가장 작은 경우는 4입니다.
이처럼 숫자 N과 number가 주어질 때, N과 사칙연산만 사용해서 표현 할 수 있는 방법 중 N 사용횟수의 최솟값을 return 하도록 solution 함수를 작성하세요.

##### 제한사항

- N은 1 이상 9 이하입니다.
- number는 1 이상 32,000 이하입니다.
- 수식에는 괄호와 사칙연산만 가능하며 나누기 연산에서 나머지는 무시합니다.
- 최솟값이 8보다 크면 -1을 return 합니다.

##### 입출력 예

| N    | number | return |
| ---- | ------ | ------ |
| 5    | 12     | 4      |
| 2    | 11     | 3      |

##### 입출력 예 설명

예제 #1
문제에 나온 예와 같습니다.

예제 #2
`11 = 22 / 2`와 같이 2를 3번만 사용하여 표현할 수 있습니다.





## 동적계획법 (Dyanamic Programming)

> 주어진 최적화 문제를 재귀적인 방식으로 보다 작은 부분으로 나누어 부분 문제로 풀어, 이 해를 조합 하여 전체 문제의 해답을 구하는 방식
>
> 알고리즘에 진행에 따라 **탐색해야 할 범위를 동적으로** 결정 함으로써 탐색 범위를 한정할 수 있음



## 적용 사례

### 피보나치 수열

#### 재귀 함수

피보나치 수열을 재귀 함수로 구현 한다면? 복잡도가 지수함수형태를 띄게 된다

![image-20210424112014308](https://tva1.sinaimg.cn/large/008i3skNgy1gpumbmkb6aj30pm09w40a.jpg)

#### 동적 계획법

부분 문제(`f(0)=0`, `f(1)=1`)로 풀어 이들을 조합 한다. 복잡도는 선형함수의 형태로 나타난다.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gpumcv1cbcj30jb0arq4s.jpg" alt="image-20210424112125554" style="zoom:50%;" />

### Knapsack 문제

가장 높은 값을 가지도록 물건을 골라 배낭에 담으시오. [배낭 문제 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%EB%B0%B0%EB%82%AD_%EB%AC%B8%EC%A0%9C)



## 문제의 해결

### 동적계획법으로 설계

N을 **한 번** 사용해서 만들수 있는 수(들) -> 1

N을 **두 번** 사용해서 만들수 있는 수(들) -> 2

N을 **세 번** 사용해서 만들수 있는 수(들) -> 3

이를 반복하다가 내가 원하는 수가 나타나면 이를 반환 한다.



### 예제

`N = 5` 일때,  1번 사용해서는 1개의 숫자만 만들 수 있다. 2번 사용 했을 때는, `55`와 `1개를 사용해서 만들 것을 사칙 연산`한 것으로 각각 만들 수 있다.

![image-20210424112826721](https://tva1.sinaimg.cn/large/008i3skNgy1gpumk5u56fj313u0cmabd.jpg)

![image-20210424113053357](https://tva1.sinaimg.cn/large/008i3skNgy1gpun6ws5o0j311i0h8t9c.jpg)

![image-20210424113122357](https://tva1.sinaimg.cn/large/008i3skNgy1gpumn7vc4ij31120igae7.jpg)

이를 일반화 하면 다음과 같다.

![image-20210424113347536](https://tva1.sinaimg.cn/large/008i3skNgy1gpumpq7ge1j31370howhb.jpg)



## 코드 구현

```python
# N이 1~ 9까지
dp = [set() for _ in range(1,9)]

# i개 만큼 만들수 있는 숫자 추가
for i ,x in enumerate(dp, start=1):
    x.add(int(str(N)*i))


for i in range(1, len(dp)):
    # i=1,
    for j in range(i):
        for op1 in dp[j]:
            for op2 in dp[i-j-1]:
                dp[i].add(op1+op2)
                dp[i].add(op1-op2)
                dp[i].add(op1*op2)
                if op2 != 0:
                    dp[i].add(op1//op2)

    if number in dp[i]:
        answer = i + 1
        break
else:
    answer = -1


print(answer)
```



### 더 나은 풀이

`op1`과 `op2`를 구하는데 `itertools`의 `product` 메소드를 사용해서 구현 하였다.

```python
from itertools import product

def solution(N, number):
    dp = [set() for _ in range(8+1)]

    for i in range(1, 8+1):
        dp[i].add(int(str(N)*i))
        for j in range(1, i):
            for op1, op2 in product(dp[j], dp[i-j]):
                dp[i].add(op1+op2)
                dp[i].add(op1-op2)
                dp[i].add(op1*op2)
                if op2 != 0:
                    dp[i].add(op1//op2)
        if number in dp[i]:
            return  (i)

    return -1
```

## 기록 사항

- `enumerate`(*iterable*, *start=0*)
  - `enumerate` 함수에 start인자가 있는지 처음 알았다. start를 쓰지 못해서 어려운적이 많았는데 꼭 기억하고 다음 부터 써야 겠다
- `dict.add()`, `dict.get()`
  - dict에 다양한 메소드가 있는 걸 을 배웠다. 보통 리스트를 많이 사용해서 append 메소드만 원소 추가가 가능 한 줄 알 고있었는데 `add()` 로도 가능하고, 추가적으로 get() 메소드를 사용해서 key에 접근할 수 있다. 이때 디폴트 값을 설정 해주면 value가 없을때 디폴트 값을 가져오고 디폴트를 설정 안해주면 None을 반환 한다.