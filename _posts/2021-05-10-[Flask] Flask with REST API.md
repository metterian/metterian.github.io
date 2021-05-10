---
layout: post
title: "[Flask] Flask with REST API"
author: "metterian"
tags: 프로그래머스 Flask Python
---


# Flask로 REST API 사용하기

## API

> API는 애플리케이션 소프트웨어를 구축하고 [통합](https://www.redhat.com/ko/topics/integration)하기 위한 정의 및 프로토콜 세트로, 애플리케이션 프로그래밍 인터페이스(Application Programming Interface)를 나타냅니다.

즉, 풀어 설명하자면 프로그램들이 서로 상호 작용하는 것을 도와주는 **매채체** 를 뜻한다. 다음 그림과 같이 식당에 점원이 존재 한다고 해보자. 점원은 고객(Client)로 주문 요청(Request)를 받아 요리사(Server)에게 전달 해준다. 중간에서 점원과 같은 역확을 하는 것을 API라고 한다.

![API, 쉽게 알아보기](https://tva1.sinaimg.cn/large/008i3skNgy1gqdd75lzk2j30hr0hg3zh.jpg)

<br>

## Think RESTful!

Representational State Tranfer의 약자로써, 웹 서버가 요청을 응답하는 방법론 중 하나이다. 데이터가 아닌, **자원(Resource)**의 관점으로 접근 하는 방법론을 말한다.


### HTTP URI

HTTP URI는 자원 명시를 위해 사용된다. 어떠한 정보를 요청(Request) 할 때, 대상자 혹은 대상이 있어야 한다. 그 대상자의 Identity를 URI라고 한다. 이 URI에 속하는 것 중에 하나가 우리가 알고 있는 URL 이다.

### HTTP Method

HTTP Method는 해당 자원에 대한 CRUD를 진행 하기 위해서 사용한다. 우리가 요청을 할 때, 이미 정해진 프로토콜 즉, 메소드(방법)이 존재한다. GET, POST, PUT 등이 있다.

<br>

### 자원으로 바라보기

우리는 다음 표과 같이 각 HTTP Method들의 요청을 하나의 자원이라고 바라 보게 된다. 이렇게 자원으로 바라보는서 갖게 되는 이점은 현재 주소가 다 동일한 대상(\order)를 가리키고 있다. 즉, 이 **한 자원에 대해서 다른 메소드를 적용** 하는 것이다.

| HTTP Method | Resource |
| :---------: | :------: |
|     GET     |  /order  |
|    POST     |  /order  |
|     PUT     |  /order  |
|   DELETE    |  /order  |

<br>

### Stateless

REST API의 또 다른 특징 중에 하나는 바로 "Stateless"이다. 이 말의 의미는 Client가 Context를 서버에서 유지하지 않는다 라는 의미 이다. 다음 그림과 같이 한 서버에 여러 명의 Client가 존재 하다고 가정해보자. A가 서버에게 보내 요청이나, B가 서버에게 보낸 요청은 모두 동일하게 값을 반환 해줘야 한다.

![image-20210510174602851](https://tva1.sinaimg.cn/large/008i3skNgy1gqdfdzck4zj30gv0a174v.jpg)

단, 이와 반대의 경우는 무엇이 있을까? 바로 로그인의 경우이다. 만약 서버가 로그인 정보를 가지고 있다면 각각 Client 마다 다른 정보를 제공해줘야 할 것이다. 하지만, REST API 이렇게 하지 않고 각 Client를 독립적인 대상으로 바라보고 독립된 결과를 제공해준다.

<br>

#### 예시

가령,  다음의 예시를 살펴 보자.

- `POST/shoes` 는 새로운 정보를 생성한다.
- `GET/shoes` 는 DB(server)에서 shoes가 있는지 확인 후 해당 자원을 반환한다.

이렇게 메소드가 실행되면, 서버 입징에서 아이템을 `GET`하기 위해서 이전에 POST를 진행 할 필요가 없다. 없으면 에러를 반환하고 있으면 그냥 반환 하면 된다. 즉, 한 자원에 대해서 각 메소드들은 독립적으로 동작 하고 있는 것이다.




<br>
<br>

### REST API 예제 - Coffee Shop Menu API 구축

#### 기본 틀

다음과 같이 Flask의 기본 세팅을 해줍니다. 기본 홈 세팅으로 HTTP body에 "hello world"를 반환하는 함수를 작성 해 줍니다.

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_flask():
    return "Hello World!"

if __name__ == '__main__':
    app.run()
```

<br>

#### 메뉴

다음과 같이 메뉴가 주어졌다고 가정 해보자

```python
menus = [
    {"id": 1, 'name':"Espresso", "price": 3800},
    {"id": 1, 'name':"Americano", "price": 4100},
    {"id": 1, 'name':"CafeLatte", "price": 4600},
]
```

<br>

#### 추가 모듈 불러오기

해당 예제를 구현 하기위해서는 `jsonify`, `request` 이 추가적으로 필요하다.`jsonify` 는 딕셔너리 데이터 타입을 json 포맷으로 바꿔주는 모듈이고, `request`는 HTTP 통신을 위해 사용되는 모듈이다.

```python
from flask import Flask, jsonify, request
```

<br>

#### GET /menus 구현

GET 메소드는 자료를 가져 올 때 사용된다. 이를 코드로 구현 하면 다음과 같다. menus가 리스트 자료형이기 때문에 이를 딕셔너리에 담아서 jsonfiy를 사용하여 json 자료형으로 변환 해준다.

```python
# GET /menus
@app.route('/menus')
def get_menus():
    return jsonify({"menus": menus})
```

<br>

#### POST /menus 구현

POST 메소드는 자료를 자원에 추가 할 때 사용된다.  이 구현에서는 request 요청이 json 자료형으로 요쳥된다는 가정하에 코드를 작성 하였다.

```python
@app.route('/menus', method=['POST'])
def create_menu(): # request가 JSON이라고 가정
    # 전달 받은 자료를 menus를 자원에 추가
    request_data = request.get_json() # request 요청된 데이터를 가져온다.
    new_menu = {
        "id": 4,
        "name": request_data['name'],
        "price": request_data['price']
    }
    menus.append(new_menu)
    return jsonify(new_menu)
```


<br>


## POSTMAN

### '/' 확인

**Postman** 을 통해서 API를 간단하게 테스트 해 볼 수 있습니다. Postman에 접속하여 새로운 collection을 만들고 테스트 중인 flask의 주소와 포트 번호를 입력하고 `Send` 를 클릭 해 봅시다.

![image-20210510200538042](https://tva1.sinaimg.cn/large/008i3skNgy1gqdjf86nesj30n806caaj.jpg)

그럼 다음과 같이 우리가 코딩 해논 결과를 확인 할 수 있습니다.

![image-20210510200719175](https://tva1.sinaimg.cn/large/008i3skNgy1gqdjgzg4yjj30ez074aa6.jpg)

### GET Menus

다음과 같이 `http://127.0.0.1:5000/menus` 를 입력하고, 출력 결과를 확인 해 볼 수 있습니다.

![image-20210510201136228](https://tva1.sinaimg.cn/large/008i3skNgy1gqdjlg0rgpj30ea0cx750.jpg)

### POST Menus

![image-20210510201502010](https://tva1.sinaimg.cn/large/008i3skNgy1gqdjp08seqj30mw075dge.jpg)

위 그림과 같이 POST로 설정을 바꾸고 Body를 클릭하여 Sever에 JSON 내용을 REQUEST 해봅시다.

![image-20210510201608054](https://tva1.sinaimg.cn/large/008i3skNgy1gqdjq5e5psj30jz05wdg5.jpg)

위 그림 처럼 "200 OK" 가 뜨고 정상적으로 실행 된 것을 확인 해 볼 수 있습니다. 이 다음 GET를 요청하면 다음과 같이 POST로 전송한 데이터가 출력 됩니다.

![image-20210510201727521](https://tva1.sinaimg.cn/large/008i3skNgy1gqdjrke3ryj30jr0dhab2.jpg)


<br>

### 404 에러

만약 우리가 제공하지않은 자원에 접속하려 한다면 다음과 같이 404에러를 출력합니다.

![image-20210510201009034](https://tva1.sinaimg.cn/large/008i3skNgy1gqdjvwg91dj30mk07q0tg.jpg)



<br>

### 자원 저장하기

Save 버튼을 눌러 다음과 같이 저장 해놓으면 다양한 테스트를 할 때 유용하게 사용 할 수 있습니다.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdjn0srzqj30dd0j9q3t.jpg" alt="image-20210510201307132" style="zoom: 67%;" />

<br>

### 데이터 베이스

이렇게 POST를 통해서 서버에 데이터를 입력하고 서버를 껐다가 키면 어떻게 될까요? 아쉽게도 서버의 내용이 삭제 되고 맙니다. 때문에, 서버에는 이러한 데이터를 저장 할 수 있는 데이터베이스가 필요하고 이를 Flask에 연동 시켜 놔야 합니다. 이는 다음 포스팅에서 살펴 보겠습니다.
