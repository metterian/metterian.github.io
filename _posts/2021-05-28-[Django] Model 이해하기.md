---
layout: post
title: "[Django] Model 이해하기"
author: "metterian"
tags: 
---
## 들어가며

이전까지 View를 살펴보았습니다. template를 활용해서 view의 RESPONSE를 처리하는 것을 살펴 보았는데 이번 시간에는 model를 활용해서 view의 RESOPONSE를 처리하는 것을 살펴 보겠습니다.

![image-20210519173317130](https://tva1.sinaimg.cn/large/008i3skNgy1gqntljmsk9j30fu08iq3u.jpg)

Model을 살펴보기 이전에 데이터 베이스를 먼저 살펴 보겠습니다. 데이터 베이스의 가장 큰 특징은 **구조화** 한다는 것이 가장 큰 특징 입니다. 그러한 구조화된 데이터베이스 중에서 가장 대표적인적인 것은 관계형 데이터베이스입니다. 즉, 테이블을 통해서 데이터베이스를 다루는 것을 말합니다. 이것을 다루는 언어를 SQL이라고 합니다. 하지만, 장고에서는 이 SQL를 이용하지 않고도 테이블을 다룰 수 있습니다. 이를 ORM이라고 합니다. 이 ORM을 활용에서 어떻게 데이터 베이스를 활용하는 지에 대해 이번 포스팅에서 다루고자 합니다.

<br>

## Model 살펴보기

### Class 선언

기존에 만든 App을 살펴 보면 자동적으로 `models.py` 라는 파일이 생성 되었습니다. 이를 활용해서 장고에서는 클래스단위로 모델을 만들어 줄 수 있습니다. 즉, `models.Model` 상속 받아 다음과 같이 클래스를 구성하게 됩니다.

```python
from django.db import models

# Create your models here.
class <모델 이름> (models.Model):
```

<br>

### Field 선언

`Coffee` 라는 클래스를 선언 했다고 가정 해 봅시다. 이렇게 클래스를 선언하고 그 안에 한 개의 변수를 선얼 할 때마다. 한 개의 field(=attribute)가 생성되게 됩니다. 이때 이 FieldType은 문자열, 숫자, 논리형, 시간/날짜 ..  형으로 구성 할 수 있습니다. 대표적인 타입형을 다음과 같습니다.

- 문자열: `CharField`
- 숫자: `IntegerField`, `SmallIntegerField`, ...
- 논리형: `BooleanField`
- 시간/날짜: `DatetimeField`

```python
class Coffee(models.Model):
    field1 = models.FieldType()...
    field1 = models.FieldType()...
    field1 = models.FieldType()...
```



#### FieldType 인자

FieldType의 인자로 `default`와 `null`등을 인자로 전달 할 수 있습니다. `defualt`는 기본값으로 어떤 값을 지정할 수 있고, `null`은 빈 값이 들어가도 되는지 여부를 설정하는 것입니다. 또한 `max_length`는 값의 최대 길이를 설정 할 수 있습니다.

```python
class Coffee(models.Model):
    name = models.CharField(default="", null=False, max_length=30)
    price = models.IntegerField(default=0)
    is_ice = models.BooleanField(default=False)
```



## Admin 살펴보기

장고의 가장 큰 특징은 어떠한 모델이 있을때 이를 관리자 페이지에서 관리를 해줄 수 있습니다. 이를 활성화 해주기 위해 `admin.py` 를 수정해줍니다.

수정하기 전에, 모델의 정보를 불러와야 하기 때문에 다음과 같이 상단에 모델의 정보를 import 해줍니다.

```python
from .models import Coffee
```

이렇게 추가한 후 다음과 같이 한 줄을 더 추가 해줍니다. 이렇게 한 줄을 추가해 주면 관리자 페이지에서 Coffe 모델을 관리 할 수 있게 됩니다.

```python
admin.site.register(Coffee)
```

<br>

<br>

## 데이터 베이스 살펴보기

이렇게 위의 내용들을 수정하고 장고의 Admin 페이지를 들어가 보면 다음과 같은 에러를 만나게 됩니다. `no such table: homepage_coffee` 를 확인해 볼 수 있으며, 우리가 아직 장고의 DB를 Migration 해주지 않았기 때문입니다. 

![image-20210519175614692](https://tva1.sinaimg.cn/large/008i3skNgy1gqnu9dzbllj30su0avmzp.jpg)

### Migration

기존에는 `python manage.py migrate` 를 실행하여 Migration를 시켜 주었습니다. 기존에는 우리가 만든 DB가 존재하지 않았기 때문에 실행 하였던 것입니다. 때문에, 우리는 새로운 DB를 반영하기 위해서 다음과 같이 CLI를 입력해 줍니다. 

```bash
$ python3 manage.py makemigrations
```

이를 실행하면 다음과 같은 결과를 확인 할 수 있습니다.

```bash
Migrations for 'homepage':
  homepage/migrations/0001_initial.py
    - Create model Coffee
```

이후, 이전과 같이 다음 명령어를 실행해서 Migration을 진행 해 줍니다.

```bash
python3 manage.py migrate
```

<br>

## 대표값 설정하기

위와 같이 설정하고 Admin 페이지에 들어가서 데이터를 만들고 나면 다음과 같이 Object라는 이름으로 데이터가 생성 됩니다. 우리는 이것이 어떤 데이터 인지 확인 할 수 없기 때문에 이를 대표 할 수 있는 Field를 선택해서 표시 해 줘야 합니다.

![image-20210519180411786](https://tva1.sinaimg.cn/large/008i3skNgy1gqnuhobomyj307o02zweg.jpg)

### `__str__(self)`

`__str__(self)` 이 메소드는 클래스를 어떠한 방식으로 표시 해줄지를 설정 할 수 있는 메소드 입니다. 이를 코드상에서 다음과 같이 작성 해 줍니다. name이라는 field를 대표값으로 설정해서 이를 표기 해준다는 의미 입니다.

```python
class Coffee(models.Model):
    def __str__(self) -> str:
        return self.name
    name = models.CharField(default="", max_length=30)
    price = models.IntegerField(default=0)
    is_ice = models.BooleanField(default=False)
```



실제 Admin 페이지에서 이를 확인 하면 다음과 같이 확인 할 수 있습니다.

![image-20210519180739000](https://tva1.sinaimg.cn/large/008i3skNgy1gqnul9bzobj305e034wed.jpg)

