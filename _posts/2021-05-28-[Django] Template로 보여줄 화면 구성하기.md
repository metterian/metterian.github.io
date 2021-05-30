---
layout: post
title: "[Django] Template로 보여줄 화면 구성하기"
author: "metterian"
tags: Django
---
## 들어가며

보여줄 화면을 어떻게 구성하면 좋을지를 한번 살펴보도록 하겠습니다 이전에 View해서는 들어온 요청에 대한 처리를 담당하는 함수을 작성한다고 했는데요. 이 과정에서 보여줄 문서 등을 이거 Template라고 합니다. 이에 해당하는 것들이 바로 html, css, 자바스크립트가 있습니다.

View는 html과 css 그리고 자바스크립트를 렌더링 하는 과정에서 동적인 웹사이트를 만들도록 해줍니다. 왜 템플릿이 필요한지, 또 템플릿을 어떻게 만들면 되는지 한번 알아보도록 하겠습니다.

<br>

## Template

### `view.py`  살펴보기

기존의 `view.py` 에 다음과 같이 HTML 마크업을 추가 하여 결과를 확인 해봅니다.

![image-20210518145140433](https://tva1.sinaimg.cn/large/008i3skNgy1gqmjb4aqaqj30fe049mxf.jpg)

위의 그림과 같이 HTML 마크업이 잘 적용 되었습니다. 하지만, HTML 파일의 크기가 매우 큰 상황이면 어떻해야 할까요? Django는 이러한 상황을 대비해서 `render` 함수를 제공해 줍니다.

<br>

### `render` 함수

> [`render()`](https://docs.djangoproject.com/ko/3.2/topics/http/shortcuts/#django.shortcuts.render) 함수는 request 객체를 첫번째 인수로 받고, 템플릿 이름을 두번째 인수로 받으며, context 사전형 객체를 세전째 선택적(optional) 인수로 받습니다. 인수로 지정된 context로 표현된 템플릿의 [`HttpResponse`](https://docs.djangoproject.com/ko/3.2/ref/request-response/#django.http.HttpResponse) 객체가 반환됩니다.

위의 설명된 render 함수를 사용하연 view.py를 다음과 같이 수정 해 줍니다.

```python
from django.shortcuts import HttpResponse, render

# Create your views here.
def index(request):
    return render(request, 'index.html')
```

이렇게 render 함수를 사용하므로써 크기가 큰 html 파일을 HTTP Response 프로토콜 형식으로 전송할 수 있게 되었습니다.

<br>



### template 디렉토리 생성

`render` 함수에서 사용할 HTML 파일을을 저장하기 위해서 app 폴더에 template라는 폴더를 추가 해 줍니다. 이후, 이곳에 `index.html` 파일을 만들어 추가 해줍니다.

```bash
homepage
├── __init__.py
├── admin.py
├── apps.py
├── migrations
├── models.py
├── template # 폴더 추가
│   └── index.html
├── tests.py
└── views.py
```

<br>

`index.html`은 다음과 같이 코드를 추가해 줍니다.

```html
<!DOCTYPE html>
<html>
    <head>
        <title>
            Python Django example
        </title>
    </head>

    <body>
        <h1>Title</h1>
        <p>test</p>
    </body>
</html>
```

<br>

### Template 파일 명시

template 파일들은 Python 파일이 아니기 때문에 import 문들을 작성 할 수 없습니다. 때문에, 이를 `setting.py` 에 이를 명시 해줘야 합니다. setting.py 파일에 DIR 인자에 다음의 값들은 입력 해 줍니다.

```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(BASE_DIR, 'homepage', 'template') # tempate 주소
        ],
    }
]
```



### Render  살펴보기 - 동적 페이지

```python
from django.shortcuts import HttpResponse, render

# Create your views here.
def index(request):
    return render(request, 'index.html', {})
```

위에서 작성한 render 함수의 경우 어떠한 내용을 보여준다의 의미 보다는 어떤 데이터를 바탕으로 HTML의 내용을 완성한다는 느낌이 더 가깝습니다. 때문에, render 함수의 인자로 주어진 데이터를 활용하여 동적인 HTML 페이지를 만들어 낼 수 있습니다.

<br>

다음과 같이 number라는 변수를 추가하고 이를 render 함수에 딕셔너리에 다음과 같이 추가 해 줍니다. 여기서 `my_num` 은 HTMLP 템플릿 언어로 사용됩니다.

```python
def index(request):
    number = 50
    return render(request, 'index.html'. {"my_num":number})
```

<br>
다시, `index.html` 파일을 열어 다음과 같이 수정 해 줍니다. `{{ my_num}}` 를 다음과 같이 추가 해줍니다. 우리가 Python 모듈을 사용하기 위해 import 문을 사용하듯이 html 언어에서 모듈, 변수를 사용하기 위해서 import 작업을 render 함수에 딕셔너리 인자를 추가하여 import와 비슷한 기능하도록 도와줍니다.

```html
<!DOCTYPE html>
<html>
    <head>
        <title>
            Python Django example
        </title>
    </head>

    <body>
        <h1>Title</h1>
        <p>test</p>
        <p>{{ my_num}} # 추가된 템플릿 언어</p>
    </body>
</html>
```

<br>

다음과 같이 결과를 확인 해 볼 수 있습니다. 기존에는 html 파일에 작성한 내용만을 표기 할 수 있었습니다. 이를 정적인 페이지라고 하는데요. 하지만, Django의 template 언어를 사용하여 인위적으로 수치를 변화를 주어 동적인 페이지를 만들 수 있게 되었습니다.

![image-20210518151811683](https://tva1.sinaimg.cn/large/008i3skNgy1gqmk2o1iv4j305m057mx8.jpg)

<br>

## Template Filter

### Length

다음과 같이 `view.py` 를 수정 해 줍니다.

```python
from django.shortcuts import HttpResponse, render

# Create your views here.
def index(request):
    name = 'Michael'
    return render(request, 'index.html', {"my_name":name})
```



이후, Html 파일에 `{% raw %}{{ my_name | length}} {% endraw %}` 를 추가해 줍니다. 여기서 `|` 는 filter를 의미하고 뒤에 length를 쓰게 되면 이때의 문자의 길이을 반환하게 됩니다.

```html
<!DOCTYPE html>
<html>
    <head>
        <title>
            Python Django example
        </title>
    </head>

    <body>
        <h1>Title</h1>
        <p>test</p>
        <p>{{ my_name | length}}</p>
    </body>
</html>
```

![image-20210518152353336](https://tva1.sinaimg.cn/large/008i3skNgy1gqmk8ldnb9j305505b3yk.jpg)

### Upper

upper filter를 사용하면 문자열을 모두 대문자화 하여 출력하게 됩니다.

![image-20210518152445399](https://tva1.sinaimg.cn/large/008i3skNgy1gqmk9hqv03j30530553yf.jpg)

<br>

## Template Tag

`{% raw %}{% tag .. %}{% endraw %}` ,  `{% raw %}{% endtag .. %}{% endraw %}`로 작성하여 사용합니다.

### For tag

`{% raw %}{% for a in b %}{% endraw %}` 구조로 사용. 다음과 같이 수정하여 사용할 수 있다.

```python
def index(request):
    nums = [1,2,3,4,5]
    return render(request, 'index.html', {"my_list":nums})
```

```html
{% raw %}
{% for element in my_list %}
	<p> {{ element }} </p>
{% endfor %}{% endraw %}
```

<br><br>

### IF Tag

```bash
{% raw %}
{% for element in my_list %}
	{% if element|divisibleby:"2" %}
    	<p> {{ element }} </p>
	{% endif %}
{% endfor %}
{% endraw %}
```

