---
layout: post
title: "[Django] View로 Request 다루기"
author: "metterian"
tags: 
---
## 들어가며

장고 App에 존재하는 view는 장고의 MVP 패턴에서 중추적인 역활을 지닙니다. 사용자 REQUEST로 부터 분석된 url을 분석하여 전달 받은 view에서는 이를 model(DB)과 template를 활용하여 사용자에게 RESPONSE를 보내게 됩니다. 이때 어떻게 view가 REQUEST를 처리 하는지 한 번 알아 보겠습니다.

<br>

## `view.py` 살펴보기

`view.py` 이전에 만들었던 app(hompage)의 디렉토리 안에 존재합니다.

```bash
homepage
├── __init__.py
├── admin.py
├── apps.py
├── migrations
├── models.py
├── tests.py
└── views.py
```



가장 간단한 view를 구현하기 위해서는 함수를 통해 구현해야 합니다. 여기서는 `index` 라는 함수를 만들어 보려고 합니다. request를 인자로 받아 HTTP Respones 프로토콜로 값을 전송하는 함수를 다음과 같이 작성 할 수 있습니다. 

```python
from django.shortcuts import HttpResponse, render

# Create your views here.
def index(request):
    return HttpRespones('Hello World!')
```



## `url.py` 살펴 보기

어떠한 요청이든 이렇게 처리하면 안되고 각각의 종류의 따른 요청을 구분 할 줄 알아야합니다. 이는 view에서 처리하는 것이 아닌 `url.py`에서 실행이 됩니다. 단, 이 파일은 app 디렉토리에 존재하지 않고 프로젝트 폴더에 프로젝트 이름과 동일한 디렉토리에 존재 합니다. 

```bash
webproj
├── db.sqlite3
├── homepage
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations
│   ├── models.py
│   ├── tests.py
│   └── views.py
├── manage.py
└── webproj
    ├── __init__.py
    ├── __pycache__
    ├── asgi.py
    ├── settings.py
    ├── urls.py # 이 곳에 존재8
    └── wsgi.py
```



`url.py` 파일을 열어 보면 이미 다음과 같은 로직이 존재 합니다. admin이라는 요청이 들어 왔을때, 이를 핸들링하는 로직입니다.

```python
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
```



이름 다음과 같이 우리가 구현하고자 하는 로직을 추가해 줍니다. 첫 페이지에 들어 갔을때의 요청을 `index` 함수로 처리하기 위해서 다음과 같이 작성해 줍니다. path에서 index 함수를 사용하기 위해서는 hompage에 있는 모듈 view를 불어와야 하는 점을 기억 해야 합니다. 

```python
from django.contrib import admin
from django.urls import path
from homepage.views import index # view의 함수 import

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index ), # 127.0.0.1
]
```

<br>

### Installed App 추가

`setting.py`에 `startapp` 을 통해서 만들었던 homepage를 Installed App에 추가해 줍니다. 

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'homepage' # 추가된 App
]
```





## 관리자 계정 생성

Django의 Admin 계정을 생성하기 위해서 다음과 같은 명령어를 실행 하면 오류를 만나게 됩니다. 

```bash
$ python3 manage.py createsuperuser
```

```bash
...
django.db.utils.OperationalError: no such table: auth_user
```

<br>

이렇게 오류가 발생하는 이유는 Django 디렉토리가 생성된때 자동으로 생성된 DB 파일들이 현재의 DB에 반영되지 못했기 때문입니다. 이를 반영 시켜주기 위해서 `migrate` 함수를 사용해서 이를 반영 시켜 줍니다.

```bash
python3 manage.py migrate
```

```bash
Applying contenttypes.0001_initial... OK
Applying auth.0001_initial... OK
Applying admin.0001_initial... OK
Applying admin.0002_logentry_remove_auto_add... OK
...
```

그러면 위와 같이 migrate가 완료된 결과를 확인 할 수 있습니다. 