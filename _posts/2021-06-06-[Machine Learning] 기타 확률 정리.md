---
layout: post
title: "[Machine Learning] 기타 확률 정리"
author: "metterian"
tags: PRML 확률과통계
---
## 기댓값(Expectation)

- 기댓값: 확률분포 $p(x)$에서 함수 $f(x)$의 평균값
- 이산확률분포(discrete distirubution): $\mathbb{E}[f]=\sum_{x} p(x) f(x)$
- 연속확률분포(continous distiribution): $\mathbb{E}[f]=\int p(x) f(x) \mathrm{d} x$

모든 경우에 이러한 적분을 통해서 기댓값을 구하기 어려우므로, 확률 분포로 부터 $N$개의 샘플을 추출해서 기댓값을 근사 할 수 있다.

$$
\mathbb{E}[f] \approx \frac{1}{N} \sum_{n=1}^{N} f\left(x_{n}\right)
$$

### 확률 변수가 한 개가 아닌 경우

#### 여러개의 확률변수로 주어진 경우

이 경우, $x$에 대한 summation을 구한 뒤 임으로 $y$에 대한 변수만 남게 된다. 그러므로 다음 식은 $y$에 대한 함수 함수 이다.

$$
\mathbb{E}_{x}[f(x, y)]=\sum f(x, y) p(x)
$$

#### 결합 확률 변수인 경우

$$
\mathbb{E}_{x, y}[f(x, y)]=\sum \sum f(x, y) p(x, y)
$$

### 조건부 기댓값(conditional expectation)

$$
\mathbb{E}_{x}[f \mid y]=\sum f(x) p(x \mid y)
$$

<br>

## 분산(Variance), 공분산(Covariance)

$f(x)$의 분산(variance)이란, $f(x)$의 값들이 기댓 $\mathrm{E}[f]$ 으로 부터 흩어져 있는 정도를 나타내며 다음과 같은 식으로 표현된다.

$$
\begin{aligned}
\operatorname{var}[f] &=\mathbb{E}\left[(f(x)-\mathbb{E}[f(x)])^{2}\right]=\mathbb{E}\left[f(x)^{2}\right]-\mathbb{E}[f(x)]^{2} \\
\operatorname{var}[x] &=\mathbb{E}\left[x^{2}\right]-\mathbb{E}[x]^{2}
\end{aligned}
$$

### 두 개의 확률변수 $x,y$에 대한 공분산

$$
\begin{aligned}
\operatorname{cov}[x, y] &=\mathbb{E}_{x, y}[\{x-\mathbb{E}[x]\}\{y-\mathbb{E}[y]\}] \\
&=\mathbb{E}_{x, y}[x y]-\mathbb{E}[x] \mathbb{E}[y]
\end{aligned}
$$

### $x,y$가 각각 확률변수의 벡터라고 할 때

$$
\begin{aligned}
\operatorname{cov}[\mathbf{x}, \mathbf{y}]=& \mathbb{E}_{\mathbf{x}, \mathbf{y}}\left[\{\mathbf{x}-\mathbb{E}[\mathbf{x}]\}\left\{\mathbf{y}^{T}-\mathbb{E}\left[\mathbf{y}^{T}\right]\right\}\right] \\
=& \mathbb{E}_{\mathbf{x}, \mathbf{y}}\left[\mathbf{x} \mathbf{y}^{T}\right]-\mathbb{E}[\mathbf{x}] \mathbb{E}\left[\mathbf{y}^{T}\right] \\\\
& \operatorname{cov}[\mathbf{x}] \equiv \operatorname{cov}[\mathbf{x}, \mathbf{x}]
\end{aligned}
\\
\text{단, 여기서 $\mathbf{x}$는 확률 벡터를 의미한다. }
$$

<br>

## 빈도주의 vs. 베이지안

확률을 해석하는 두 가지 다른 관점이 존재합니다.

- 빈도주의: 반복가능한 사건들의 빈도수에 기반
- 베이지안: 불확실성을 정량적으로 표현

### 반복가능하지 않은 사건일 경우

북극 얼음이 이번 세기말까지 녹아 없어질 확률? 우리가 이미 알고 있는 정보(얼음이 녹고 있는 속도)에 근거해 확률을 정량적으로 나타낼 수 있고 새로 수집하는 정보에 따라 확률을 업데이트 할 수 있습니다. 즉, 베이지안은 반복 가능하지 않은 사건을 정량적인 확률로 표현 할 수 있는 장점이 있습니다.

<br>

### 모델의 파라미터 사용

모델의 파라미터 $W$ (예를 들어 다항식 곡선 근사문제에서 가중치 $W$)에 대한 우리의 지식을 확률적으로 나타내고 싶다면?

- $W$에 대한 사전지식 $p(w) \ \Rightarrow$ 사전확률(prior)

- 새로운 데이터 $$\mathcal{D}=\left\{t_{1}, \ldots, t_{N}\right\}$$ 를 관찰하고 난 뒤의 조건부확률 $p(\mathcal{D} \mid \mathbf{w}) \Rightarrow$ 우도함수(likehood function).

  특정 $w$ 값에 대해 $D$의 관찰값이 얼마나 가능성이 있는지를 나타냄. $W$에 관한 함수임을 기억 해야 합니다.
  $$
  p(\mathbf{w} \mid \mathcal{D})=\frac{p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})}
  $$

- $p(\mathbf{w} \mid \mathcal{D}) \text { 는 } \mathcal{D}$를 관찰하고 난 뒤의 $\mathbf{w}$ 에 대한 불확실성을 표현
- 사후확률(posterior) $\propto$ 우도(likehhood) $\times$ 사전확률(prior)

반면, 빈도주의 $\mathbf{w}$ 가 고정된 파라미터이고, 최대우도와 같은 '추정자(estimator)'를 사용해서 그 값을 구합니다. 구해진 파라미터의 불확실성은 부트스트랩(bootstrap) 방법을 써서 구할 수 있습니다.

<br>

### 베이지안 관점의 장점

- 사전 확률을 모델에 포함 시킬 수 있습니다.
- 동전을 던저서 세번 다 앞면이 나왔을 때
  - 최대우도: 앞 면이 나올 확률은 1인 된다.
  - 베이지안: 극단적인 확률은 필할 수 있습니다.

<br>

## 정규 분포(Gaussian Distribution)

단일 변수 $x$ 를 위한 가우시안 분포 다음과 같습니다. 여기서 중요한 파라미터는 $\mu$와 $\sigma$ 인 것을 기억해야 합니다.

$$
\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)=\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left\{-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right\}
$$

### 정규화(Normalized)의 의미

음의 극한($-\infty$)부터 양의 극한($\infty$)까지 적분을 즉 넚이를 구했을 때 그 값이 1이 되면 이때 이 pdf는 표준화 되었다는 것을 의미 합니다.

$$
\int_{-\infty}^{\infty} \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) \mathrm{d} x=1
$$

<br>

### 정규 분포의 표준화(Standization) 증명

$$
\begin{aligned}
I &=\int_{-\infty}^{\infty} \exp \left(-\frac{1}{2 \sigma^{2}} x^{2}\right) \mathrm{d} x \\
I^{2} &=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \exp \left(-\frac{1}{2 \sigma^{2}}\left(x^{2}+y^{2}\right)\right) \mathrm{d} x \mathrm{~d} y \\
&=\int_{0}^{2 \pi} \int_{0}^{\infty} \exp \left(-\frac{1}{2 \sigma^{2}} r^{2}\right) r \mathrm{~d} r \mathrm{~d} \theta \\
&=\int_{0}^{2 \pi}\left\{-\left.\sigma^{2} \exp \left(-\frac{1}{2 \sigma^{2}} r^{2}\right)\right|_{0} ^{\infty}\right\} \mathrm{d} \theta \\
&=\int_{0}^{2 \pi} \sigma^{2} \mathrm{~d} \theta \\
&=2 \pi \sigma^{2} \\
I &=\sqrt{2 \pi \sigma^{2}}
\end{aligned}
$$

<br>

### 정규 분포의 기댓값(Expectation) 증명




$$
\begin{aligned}
\mathbb{E}[x]=& \int_{-\infty}^{\infty} \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) x \mathrm{~d} x \\
=& \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right\} x \mathrm{~d} x \\
=& \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{1}{2 \sigma^{2}} y^{2}\right\}(y+\mu) \mathrm{d} y \\
=& \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{1}{2 \sigma^{2}} y^{2}\right\} y \mathrm{~d} y \\
&+\mu \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left\{-\frac{1}{2 \sigma^{2}} y^{2}\right\} \mathrm{d} y \\
=& 0+\mu \cdot 1 \\
=& \mu
\end{aligned}
$$

### 정규 분포의 분산(Variance) 증명

$$
\frac{d}{d y} \int_{-\infty}^{\infty} \mathcal{N}(x \mid \mu, y) \mathrm{d} x=0
\\
\begin{array}{l}
\frac{d}{d y} \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi y}} \exp \left\{-\frac{1}{2 y}(x-\mu)^{2}\right\} \mathrm{d} x \\
=\int_{-\infty}^{\infty}\left\{\left(\frac{d}{d y} \frac{1}{\sqrt{2 \pi y}}\right) \exp \left\{-\frac{1}{2 y}(x-\mu)^{2}\right\}+\frac{1}{\sqrt{2 \pi y}}\left(\frac{d}{d y} \exp \left\{-\frac{1}{2 y}(x-\mu)^{2}\right\}\right)\right\} \mathrm{d} x \\
=-\frac{1}{2} \frac{1}{\sqrt{2 \pi}} y^{-3 / 2} \int_{-\infty}^{\infty} \exp \left\{-\frac{1}{2 y}(x-\mu)^{2}\right\} \mathrm{d} x+\frac{1}{2 y^{2}} \int_{-\infty}^{\infty}(x-\mu)^{2} \frac{1}{\sqrt{2 \pi y}} \exp \left\{-\frac{1}{2 y}(x-\mu)^{2}\right\} \mathrm{d} x \\
=-\frac{1}{2} \frac{1}{\sqrt{2 \pi}} y^{-3 / 2} \sqrt{2 \pi y}+\frac{1}{2 y^{2}} \operatorname{var}[x] \\
=-\frac{1}{2}+\frac{1}{\longrightarrow} \operatorname{var}[x]
\end{array}
\\
\operatorname{var}[x]=y=\sigma^{2}
$$

<br>

### 정규분포의 최대우도해(Maximum Likehood Solution)

$\mathbf{X}=\left(x_{1}, \ldots, x_{N}\right)^{T}$ 가 독립적으로 가우시안 분포로 부터 추출된 $N$ 개의 샘플들이라고 할때,
$$
\begin{array}{c}
p\left(\mathbf{X} \mid \mu, \sigma^{2}\right)=p\left(x_{1}, \ldots, x_{N} \mid \mu, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) \\ \\
\ln p\left(\mathbf{X} \mid \mu, \sigma^{2}\right)=-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu\right)^{2}-\frac{N}{2} \ln \sigma^{2}-\frac{N}{2} \ln (2 \pi)
\end{array}
$$

$$
\begin{aligned}
\frac{\partial}{\partial \mu} \ln p\left(\mathbf{X} \mid \mu, \sigma^{2}\right) &=\frac{\partial}{\partial \mu}\left\{-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu\right)^{2}-\frac{N}{2} \ln \sigma^{2}-\frac{N}{2} \ln (2 \pi)\right\} \\
&=-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N} 2\left(x_{n}-\mu\right) \cdot(-1) \\
&=\frac{1}{\sigma^{2}}\left\{\left(\sum_{n=1}^{N} x_{n}\right)-N \mu\right\}
\end{aligned}

\\
\mu_{M L}=\frac{1}{N} \sum_{n=1}^{N} x_{n}
$$

$$
\begin{array}{l}
y=\sigma^{2} \\
\begin{aligned}
\frac{\partial}{\partial y} \ln p(\mathbf{x} \mid \mu, y) &=\frac{\partial}{\partial y}\left\{-\frac{1}{2} y^{-1} \sum_{n=1}^{N}\left(x_{n}-\mu_{M L}\right)^{2}-\frac{N}{2} \ln y-\frac{N}{2} \ln (2 \pi)\right\} \\
&=\frac{1}{2} y^{-2} \sum_{n=1}^{N}\left(x_{n}-\mu_{M L}\right)^{2}-\frac{N}{2} y^{-1} \\
y_{M L}=\sigma_{M L}^{2}=& \frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\mu_{M L}\right)^{2}
\end{aligned}
\end{array}
$$

