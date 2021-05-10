---
layout: post
title: "[Matplotlib] python으로 멋드러진 그래프 그리기"
author: "metterian"
tags: 프로그래머스 Matplotlib Python
---
# Matlab으로 데이터 시각화하기

**데이터를 보기좋게 표현해봅시다.**

#### 1. Matplotlib 시작하기

#### 2. 자주 사용되는 Plotting의 Options
- 크기 : `figsize`
- 제목 : `title`
- 라벨 : `_label`
- 눈금 : `_tics`
- 범례 : `legend`
#### 3. Matplotlib Case Study
- 꺾은선 그래프 (Plot)
- 산점도 (Scatter Plot)
- 박스그림 (Box Plot)
- 막대그래프 (Bar Chart)
- 원형그래프 (Pie Chart)
#### 4. The 멋진 그래프, seaborn Case Study
- 커널밀도그림 (Kernel Density Plot)
- 카운트그림 (Count Plot)
- 캣그림 (Cat Plot)
- 스트립그림 (Strip Plot)
- 히트맵 (Heatmap)

## I. Matplotlib 시작하기
- 파이썬의 데이터 시각화 라이브러리
- `%matplotilb inline을 통해서 활성화


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

  

## II. Matplotlib Case Study


```python
plt.plot([1,2,3,4,5]) # 실제 plotting을 하는 함수 # y= x+ 1
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddydbtvyj30ac06waa1.jpg)



```python
plt.figure(figsize=(6,6))
plt.plot([0,1,2,3,4])
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddyc0174j30ac09x74b.jpg)




### 2차 함수 그래프 with plot()


```python
# 리스트를 이용해서 1차함수 y=x 를 그려보면
plt.plot([0,1,2,3,4])
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddyg3vn1j30ac06waa1.jpg)



```python
x = np.arange(-10, 10, 0.01) # 정의역
x[:5]
```




    array([-10.  ,  -9.99,  -9.98,  -9.97,  -9.96])




```python
plt.plot(x, x**2)
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddyjqg48j30af06w3yj.jpg)



```python
# x, y 축에 설명 추가 하기
x = np.arange(-10, 10, 0.01)

plt.xlabel('x value')
plt.ylabel("$f(x)$ value")

plt.plot(x, x**2)
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddyl4h3bj30aw07aaa4.jpg)



```python
# x,y 축의 범위를 설정하기
x = np.arange(-10, 10, 0.01)

plt.xlabel('x value')
plt.ylabel("$f(x)$ value")

# 추가
plt.axis([-5, 5, 0, 25]) #[x_min, x_max, y_min, y_max]

plt.plot(x, x**2)
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddynij5cj30ap07eq2z.jpg)



```python
# x,y 축에 눈금 수정하기

x = np.arange(-10, 10, 0.01)

plt.xlabel('x value')
plt.ylabel("$f(x)$ value")
plt.axis([-5, 5, 0, 25])

plt.xticks([i for i in range(-5,6,1)]) # x 축의 눈금 설정
plt.yticks([i for i in range(0, 10,3)]) # y 축의 눈금 설정


plt.plot(x, x**2)
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddyqdvefj30am07aq2z.jpg)



```python
# 그래프에 title 달기

x = np.arange(-10, 10, 0.01)

plt.xlabel('x value')
plt.ylabel("$f(x)$ value")
plt.axis([-5, 5, 0, 25])

plt.xticks([i for i in range(-5,6,1)]) # x 축의 눈금 설정
plt.yticks([i for i in range(0, 27,3)]) # y 축의 눈금 설정

plt.title("$y = x^2$ graph")

plt.plot(x, x**2, label="trend")
plt.legend()
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddyrr802j30at07u3ym.jpg)




## III. Matplotlib Case Study

### 꺽은선 그래프 (plot)



```python
x = np.arange(20) # 0~19
y = np.random.randint(0,22, 20) # 0~21 까지 20번 생성

x,y
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19]),
     array([ 0,  9,  1,  7, 10,  0,  6, 18,  5,  6, 20, 18,  8, 16, 17, 15, 16,
            12,  0, 14]))




```python
plt.plot(x,y)
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddytlfb9j30ai06wq34.jpg)



```python
# Extra: y축을 20까지 보이가 하고 싶다면? , y축을 5단윌 보이게 하고 싶다면?

x = np.arange(20) # 0~19
y = np.random.randint(0,22, 20) # 0~21 까지 20번 생성

plt.plot(x,y)
plt.axis([0,21, 0, 22])
plt.yticks([i for i in range(0, 21, 5)])
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddyw48nij30a806wdfy.jpg)




### 산점도 (Scatter plot)


```python
plt.scatter(x,y)
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddywtrl4j30ai06wq2t.jpg)




### Box plot

- 수치형 데이터에 대한 정도 Q1, Q2, Q3, min, max


```python
plt.boxplot((x,y))
plt.title('Box plot of $x, y$')
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddyyp6unj30ai07emx0.jpg)




### 막대 그래프 (bar plot)

- 범주형 데이터의 "값"과 그 값의 크기를 직사각형으로 나타낸 그림


```python
plt.bar(x,y) # 확률변수, 빈도
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddz04rusj30an06wmx0.jpg)



```python
# xticks를 올라를게 처리 해보기

plt.bar(x,y) # 확률변수, 빈도
plt.xticks([i for i in range(0, 21, 1)])
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddz2ht4uj30ai06w745.jpg)




#### cf) Historam

- `hist()`
- 도수분포를 직사각형의 막대 형태로 나타냈다.
- "계급" -> 그룹화로 나타낸 것이 특징: 0,1,2가 아니라 0~2까지의 범주형 데이터로 구성후 그림을 그림



```python
plt.hist(y, bins=np.arange(0,20,2))
plt.xticks(np.arange(0,20,2))
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddz4st91j30ac06wglg.jpg)




### 원형 그래프 (Pie chart)

- `pie()`
- 데이터 전체에 대한 부분의 비율을 부채꼴로 나타낸 그래프
- 다른 그래프에 비해서 **비율** 확인에 용이


```python
z = [100, 200, 300, 400]

plt.pie(z, labels=['one', 'two', 'three','four'])
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddz6qe1gj306p06f0sn.jpg)






## IV. The 멋진 그래프, Seaborn Case Study

## Seaborn Import 하기


```python
import seaborn as sns
```

### 커널 밀도 그림 (Kernel Density Plot)
- 히스토그램과 같은 연속적인 분포를 곡선과 해서 그린 그림
- `sns.kdeplot()`


```python
# in Histogram

x = np.arange(0,22,2)
y = np.random.randint(0,20, 20)

plt.hist(y, bins=x)
plt.xticks(range(0,22,2))
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddz9krs0j30ac06wglg.jpg)



```python
# kdeplot

sns.kdeplot(y, shade=True)
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddza2fi9j30aw070jrf.jpg)


히스토그램의 분포가 연속적으로 나타내어 졌다.

 

## 카운트 그림 (Count Plot)

- 범주형 column의 빈도수를 시각화 $\rightarrow$ groupby 후 도수를 하는 것과 동일한 효과
- `sns.countplot()`


```python
vote_df = pd.DataFrame({"name": ['Andy', 'Bob','Cat' ], 'vote': [True, True, False]})
vote_df
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
      <th>name</th>
      <th>vote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Andy</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cat</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# in matplotlib barplot

vote_count = vote_df.groupby('vote').count()
vote_count
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
      <th>name</th>
    </tr>
    <tr>
      <th>vote</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>1</td>
    </tr>
    <tr>
      <th>True</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(x=[False, True], height=vote_count['name'])

plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddzcrp0xj30ai06w3yd.jpg)



```python
# sns의 countplot
sns.countplot(x = vote_df['vote'])
plt.show()
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddzebledj30aw07a3yd.jpg)




### 캣그림 (Cat plot)
- 여기서 cat은 concat에서 유래
- 숫자형 변수와 하나 이상의 범주형 변수의 관계를 보여주는 함수
- `sns.catplot()`


```python
covid = pd.read_csv('data/country_wise_latest.csv')
covid.head(3)
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country/Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Active</th>
      <th>New cases</th>
      <th>New deaths</th>
      <th>New recovered</th>
      <th>Deaths / 100 Cases</th>
      <th>Recovered / 100 Cases</th>
      <th>Deaths / 100 Recovered</th>
      <th>Confirmed last week</th>
      <th>1 week change</th>
      <th>1 week % increase</th>
      <th>WHO Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>36263</td>
      <td>1269</td>
      <td>25198</td>
      <td>9796</td>
      <td>106</td>
      <td>10</td>
      <td>18</td>
      <td>3.50</td>
      <td>69.49</td>
      <td>5.04</td>
      <td>35526</td>
      <td>737</td>
      <td>2.07</td>
      <td>Eastern Mediterranean</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>4880</td>
      <td>144</td>
      <td>2745</td>
      <td>1991</td>
      <td>117</td>
      <td>6</td>
      <td>63</td>
      <td>2.95</td>
      <td>56.25</td>
      <td>5.25</td>
      <td>4171</td>
      <td>709</td>
      <td>17.00</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>27973</td>
      <td>1163</td>
      <td>18837</td>
      <td>7973</td>
      <td>616</td>
      <td>8</td>
      <td>749</td>
      <td>4.16</td>
      <td>67.34</td>
      <td>6.17</td>
      <td>23691</td>
      <td>4282</td>
      <td>18.07</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>




```python
s = sns.catplot(x='WHO Region' , y='Confirmed' , data=covid) # catplot의 kind의 default 값은 strip이기 떄문에 기본적으로 strip grip plot을 그린다.
s.fig.set_size_inches(10,6);
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddzgj1hxj30hg0bwwek.jpg)



```python
s = sns.catplot(x='WHO Region' , y='Confirmed' , data=covid, kind='violin')
s.fig.set_size_inches(10,6);
```


![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddzhh1pgj30ho0bw74f.jpg)




### 스트립그림(Strip Plot)

- scatter plot과 유사하게 데이터의 수치를 표현하는 그래프
- `sns.stripplot()`


```python
sns.stripplot(x = 'WHO Region', y= 'Recovered', data=covid)
```




![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddzk9ebsj30b707lwej.jpg)

```python
# cf) swamplot
# 겹치는 점들을 옆으로 분산하여 표시 해 준다.

s = sns.swarmplot(x = 'WHO Region', y= 'Recovered', data=covid);
```



![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddzm4wflj30b707l74f.jpg)




### 히트맵 (headmap)

- 데이터의 행렬을 색상으로 표현 해주는 그래프
- `sns.heatmap()`


```python
# 히트맵 예제
covid.corr()
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Active</th>
      <th>New cases</th>
      <th>New deaths</th>
      <th>New recovered</th>
      <th>Deaths / 100 Cases</th>
      <th>Recovered / 100 Cases</th>
      <th>Deaths / 100 Recovered</th>
      <th>Confirmed last week</th>
      <th>1 week change</th>
      <th>1 week % increase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Confirmed</th>
      <td>1.000000</td>
      <td>0.934698</td>
      <td>0.906377</td>
      <td>0.927018</td>
      <td>0.909720</td>
      <td>0.871683</td>
      <td>0.859252</td>
      <td>0.063550</td>
      <td>-0.064815</td>
      <td>0.025175</td>
      <td>0.999127</td>
      <td>0.954710</td>
      <td>-0.010161</td>
    </tr>
    <tr>
      <th>Deaths</th>
      <td>0.934698</td>
      <td>1.000000</td>
      <td>0.832098</td>
      <td>0.871586</td>
      <td>0.806975</td>
      <td>0.814161</td>
      <td>0.765114</td>
      <td>0.251565</td>
      <td>-0.114529</td>
      <td>0.169006</td>
      <td>0.939082</td>
      <td>0.855330</td>
      <td>-0.034708</td>
    </tr>
    <tr>
      <th>Recovered</th>
      <td>0.906377</td>
      <td>0.832098</td>
      <td>1.000000</td>
      <td>0.682103</td>
      <td>0.818942</td>
      <td>0.820338</td>
      <td>0.919203</td>
      <td>0.048438</td>
      <td>0.026610</td>
      <td>-0.027277</td>
      <td>0.899312</td>
      <td>0.910013</td>
      <td>-0.013697</td>
    </tr>
    <tr>
      <th>Active</th>
      <td>0.927018</td>
      <td>0.871586</td>
      <td>0.682103</td>
      <td>1.000000</td>
      <td>0.851190</td>
      <td>0.781123</td>
      <td>0.673887</td>
      <td>0.054380</td>
      <td>-0.132618</td>
      <td>0.058386</td>
      <td>0.931459</td>
      <td>0.847642</td>
      <td>-0.003752</td>
    </tr>
    <tr>
      <th>New cases</th>
      <td>0.909720</td>
      <td>0.806975</td>
      <td>0.818942</td>
      <td>0.851190</td>
      <td>1.000000</td>
      <td>0.935947</td>
      <td>0.914765</td>
      <td>0.020104</td>
      <td>-0.078666</td>
      <td>-0.011637</td>
      <td>0.896084</td>
      <td>0.959993</td>
      <td>0.030791</td>
    </tr>
    <tr>
      <th>New deaths</th>
      <td>0.871683</td>
      <td>0.814161</td>
      <td>0.820338</td>
      <td>0.781123</td>
      <td>0.935947</td>
      <td>1.000000</td>
      <td>0.889234</td>
      <td>0.060399</td>
      <td>-0.062792</td>
      <td>-0.020750</td>
      <td>0.862118</td>
      <td>0.894915</td>
      <td>0.025293</td>
    </tr>
    <tr>
      <th>New recovered</th>
      <td>0.859252</td>
      <td>0.765114</td>
      <td>0.919203</td>
      <td>0.673887</td>
      <td>0.914765</td>
      <td>0.889234</td>
      <td>1.000000</td>
      <td>0.017090</td>
      <td>-0.024293</td>
      <td>-0.023340</td>
      <td>0.839692</td>
      <td>0.954321</td>
      <td>0.032662</td>
    </tr>
    <tr>
      <th>Deaths / 100 Cases</th>
      <td>0.063550</td>
      <td>0.251565</td>
      <td>0.048438</td>
      <td>0.054380</td>
      <td>0.020104</td>
      <td>0.060399</td>
      <td>0.017090</td>
      <td>1.000000</td>
      <td>-0.168920</td>
      <td>0.334594</td>
      <td>0.069894</td>
      <td>0.015095</td>
      <td>-0.134534</td>
    </tr>
    <tr>
      <th>Recovered / 100 Cases</th>
      <td>-0.064815</td>
      <td>-0.114529</td>
      <td>0.026610</td>
      <td>-0.132618</td>
      <td>-0.078666</td>
      <td>-0.062792</td>
      <td>-0.024293</td>
      <td>-0.168920</td>
      <td>1.000000</td>
      <td>-0.295381</td>
      <td>-0.064600</td>
      <td>-0.063013</td>
      <td>-0.394254</td>
    </tr>
    <tr>
      <th>Deaths / 100 Recovered</th>
      <td>0.025175</td>
      <td>0.169006</td>
      <td>-0.027277</td>
      <td>0.058386</td>
      <td>-0.011637</td>
      <td>-0.020750</td>
      <td>-0.023340</td>
      <td>0.334594</td>
      <td>-0.295381</td>
      <td>1.000000</td>
      <td>0.030460</td>
      <td>-0.013763</td>
      <td>-0.049083</td>
    </tr>
    <tr>
      <th>Confirmed last week</th>
      <td>0.999127</td>
      <td>0.939082</td>
      <td>0.899312</td>
      <td>0.931459</td>
      <td>0.896084</td>
      <td>0.862118</td>
      <td>0.839692</td>
      <td>0.069894</td>
      <td>-0.064600</td>
      <td>0.030460</td>
      <td>1.000000</td>
      <td>0.941448</td>
      <td>-0.015247</td>
    </tr>
    <tr>
      <th>1 week change</th>
      <td>0.954710</td>
      <td>0.855330</td>
      <td>0.910013</td>
      <td>0.847642</td>
      <td>0.959993</td>
      <td>0.894915</td>
      <td>0.954321</td>
      <td>0.015095</td>
      <td>-0.063013</td>
      <td>-0.013763</td>
      <td>0.941448</td>
      <td>1.000000</td>
      <td>0.026594</td>
    </tr>
    <tr>
      <th>1 week % increase</th>
      <td>-0.010161</td>
      <td>-0.034708</td>
      <td>-0.013697</td>
      <td>-0.003752</td>
      <td>0.030791</td>
      <td>0.025293</td>
      <td>0.032662</td>
      <td>-0.134534</td>
      <td>-0.394254</td>
      <td>-0.049083</td>
      <td>-0.015247</td>
      <td>0.026594</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(covid.corr())
```




![png](https://tva1.sinaimg.cn/large/008i3skNgy1gqddzoz21dj30cz0a2mxk.jpg)
