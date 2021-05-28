---
layout: post
title: "[Django] Form으로 Template에서 Model 수정하기"
author: "metterian"
tags: 
---
## 들어가며

기존에 View를 통해서 Model을 관리하는 것을 살펴 보면 다음과 같습니다.

##### `models.py`

```python
from django.db import models

# Create your models here.
class Coffee(models.Model):
    def __str__(self) -> str:
        return self.name
    name = models.CharField(default="", max_length=30)
    price = models.IntegerField(default=0)
    is_ice = models.BooleanField(default=False)
```

homepage App에 존재하는 `models.py` 를 위 코드와 같이 작성 해주었습니다. 각 변수는 데이터베이스 상에서 attribute(=field)에 해당하고, 속성의 값을 인자로 전달 해 주었습니다. 

##### `views.py`

```python
def coffee_view(request):
    coffee_all = Coffee.objects.all()
    return render(request, 'coffee.html', {"coffee_list": coffee_all})
```

위에서 만들 model을 view에서 사용하기 위해서 `views.py` 에서 다음과 같이 코드를 수정 해주었습니다. `Coffee.objects.all()`의 의미는 model 객체의 데이터를 모두 가져온다는 의미입니다. `object` 라는 메소드를 선언하지 않음에도 이를 사용할 수 있는 이유는 `Coffee` 클래스 모델을 만들 때, `models.model` 을 상속 받았기 때문입니다. 

이후, `render` 함수를 사용해 redering 하고자 하는 HTML에 변수를 담아 rendering된 결과를 리턴 합니다. 여기서 `render` 함수는 다음곽 같은 의미가 있습니다.

```python
render(request, 'Rendering 하고자는 HTML', {HTML에서 사용할 변수 지정})
```



이렇게 작성된 코드를 통해 이번 포스팅에서는 template상에서 어떻게 Model의 내용을 수정할 수 있는 지를 살펴보겠습니다. 이를 Form을 이용해서 진행이 가능 합니다.



## Form

### Form 정의

우리가  Google Form 등에서 정해진 형식으로 데이터의 값을 채워 보는 경험을 했을 것 입니다. 이러한 형식과 마찬가지로 Django에서도 이러한 Form을 바탕으로 데이터를 채워 넣을 수 있습니다. 

![Connecting OKRs with Google Forms and Sheets - Gtmhub blog](https://tva1.sinaimg.cn/large/008i3skNgy1gqosvt89r6j31290u0dhk.jpg)

<br>

## Form 만들기

위에서 설명한 google form처럼 Django에서도 어떠한 데이터 Field에 대해서 입력 칸을 만들어 줄 수 있습니다. 

먼저, 기존에 만들었던 App에 `forms.py` 를  만들어 줍니다. 그런다음 다음과 같이 `from django import forms` 를 입력하여 form을 불어 옵니다. 그 다음 아래 코드와 같이 Meta Class를 작성 하여 줍니다. 이 때 중요한 점은 Model을 생성할 때와 마찬 가지로 Coffee Form에도 `forms.ModelForm` 을 상속 받는 다는 점입니다. 그리고 `model` 변수에는 입력 받고 싶은 모델의 객체를 담아 주고, 입력 받고 싶은 모델의 field들을 `fields` 에 담다 줍니다.

```python
from django import forms
from .models import Coffee # 모델 불러오기

class CoffeeForm(forms.ModelForm): # model form을 상속 받음
    class Meta:
        model = Coffee # 어떠한 모델에 대해 입력 값을 받을 지를 선택
        fields = ('name', 'price', 'is_ice') # 어떠한 입력값을 입력 받을 지를 선택
```



<br>

### `View.py` 에 Form 연동하기

이렇게 작성한 후, 우리가 Model을 작성하고 이를 `views.py` 에 연결 했던 것 처럼 Form도 `views.py` 에 연결을 해줘야 합니다. 먼저 우리가 사용할 Form을 불러오고 `Coffee.objects.all()` 을 객체로 사용 한 것 처럼 form도 객체로 사용 할 수 있도록 `form = CoffeeForm()` 과 같이 변수에 할당 해 줍니다. 

```python
from django.shortcuts import HttpResponse, render
from .models import Coffee
from .forms import CoffeeForm # Form 불러오기 

def coffee_view(request):
    coffee_all = Coffee.objects.all()
    form = CoffeeForm() # 객체로 사용
    return render(request, 'coffee.html', {"coffee_list": coffee_all,"coffee_form": form,}) # rendering에 form 변수 추가
```

<br>

### Template에서 Form 사용하기

Google Form 같이 우리가 만든 Template에서 사용하고자 한다면 HTML에서 `<form>` 태그를 사용해야 합니다. form을  생성한 후에 그 안에 우리가 만든 `coffee_form` 객체를 넣어 주기만 하면 됩니다. 

```html
<!DOCTYPE html>
<html>
    <head>
        <title>
            Python Django example
        </title>
    </head>

    <body>
        <h1>My Coffee List</h1>
        {% for coffee in coffee_list %}
            <p> {{coffee.name}}, {{coffee.price}} </p>
        {% endfor %}

        <form>
            {{ coffee_form.as_p }}
        </form>
    </body>
</html>
```

이렇게 작성하고 결과를 확인 하면  다음과 같습니다.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqothjfjkoj30dm0hwt9l.jpg" alt="image-20210520141504120" style="zoom:33%;" />

위 그림과 같이 입력 형태의 Form이 만들어 진 것을 확인 할 수 있습니다. 이 Form에 대한 데이터를 입력 할 수 있지만 저장하지 버튼이 없습니다. 즉, 서버로 어떠한 값을 전송 할 수 없는 것이지요.

<br>

### Save 버튼 추가하기

Save 버튼을 추가하기 위해서 다음과 같이 HTML을 수정해 줍니다. 데이터를 전송 할 수 있도록 `method="POST"` 을 form에 추가 해주고, button type을 `submit` 으로 설정 해줍니다. 이를 설명하면 button을 누르면 submit 이되고 이는 POST 형태로 전송 된다는 의미 입니다. 

```html
<form method="POST">
    {{ coffee_form.as_p }}
    <button type="submit">Save</button>
</form>
```

이렇게 작성하고 Save 버튼을 누르면 다음과 같은 오류를 만나게 됩니다.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqotocfu2mj30he05074m.jpg" alt="image-20210520142135640" style="zoom:50%;" />

이러한 오류과 발생하는 이유는 데이터를 전송할때 보안 규칙을 추가 하지 않았기 때문입니다. 이를 해결하기 위해서는 서버에서 CSRF 토큰을 form안에 `{% csrf_token %}` 를 삽입 해주면 해결이 가능 합니다. 이를 다음과 같이 수정해 줍니다.

```html
<form method="POST"> {% csrf_token %}
    {{ coffee_form.as_p }}
    <button type="submit">Save</button>
</form>
```

이렇게 작성해고 Save 버튼을 눌러도 DB에는 위의 내용이 반영되지 않습니다. 이를 반영하기 위해서 `views.py` 를 수정 해줘야 합니다. 



### View에 POST 반영하기

다음과 같은 로직으로 form의 데이터를 반영하는 view를 만들어 보도록 하겠습니다.

```bash
if request가 POST라면:
	POST를 바탕으로 Form을 완성하고
	Form이 유요하면 ->  저장
```

위의 로직을 바탕으로 `views.py` 의 내용을 수정 해 줍니다. `request.mothod == "POST"` User가  데이터를 POST로 전송하고자 할 때, POST인지 검사를 하고 이를 바탕으로 `form = CoffeeForm(request.POST)` 를 사용해서 요청된 POST를 `form` 변수에 저장합니다. 그리고 그것이 유효할 경우 이 Form을  DB에 저장합니다. `form.save()` 와 같은 메소드를 사용할 수 있는 것을우리가 이전에 Django의 Form 을 상속 받았기 때문입니다. 

```python
def coffee_view(request):
    coffee_all = Coffee.objects.all()
    
    if request.mothod == "POST":
        form = CoffeeForm(request.POST) # HTML에서 POST로 Form을 완성시킨 것을 form 변수라고 지정
        if form.is_valid(): # form의 값이 유효한지 검사
            form.save()

    form = CoffeeForm()
    return render(request, 'coffee.html', {"coffee_list": coffee_all,
                                            "coffee_form": form,})
```









