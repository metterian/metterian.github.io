---
layout: post
title: "환형 큐 (Circular Queues)"
author: "metterian"
tags: 프로그래머스 자료구조
---
# 환형 큐 (Circular Queues)

> 큐에 담을 수 있는 데이터의 양 (우리 강의에서 이용하는 용어를 가져다 쓰자면, "데이터 원소의 개수") 이 무한할 수는 없을 것입니다. 만약 큐에 담을 수 있는 원소의 개수의 상한을 미리 정하고 이를 지킬 수 있다면, 선형 배열을 이용해서 큐를 효과적으로 구현할 수 있습니다. 선형 배열의 한쪽 끝과 다른 쪽 끝이 서로 맞닿아 있는 모습 ("원형" 또는 "환형") 으로 생각하고, 큐의 맨 앞과 맨 뒤를 가리키는 (즉, 원소를 넣을 쪽의 배열 인덱스와 꺼낼 쪽의 배열 인덱스를) 기억해 두면, 데이터 원소가 빠져 나간 쪽의 저장소를 재활용하면서 큐를 관리할 수 있습니다.



## 큐의 활용

- 자료를 생성하는 작업과 그 자료를 이용하는 작업이 비동기적으로(asynchronously) 일어나는 경우

![image](https://media.vlpt.us/images/inyong_pang/post/2faad814-2d8a-45ab-9f77-5d64aa588f06/image.png)

## 환형 큐(Circular Queues)

기존의 선형 큐에서는 큐의 길이가 길어 질 수록 시간 복잡도가 길어지는 단점이 존재. 이를 해결한 것이 환형 큐. 한정된 자원 즉, 정해진 개수의 저장 공간을 빙 돌려가며 이용. 

**front**와 **rear**를 사용해가면서 데이터를 enqueue 하거나 dequeue 해서 사용

- front : 데이터를 `append`
- rear: 데이터를 `pop`

![image-20210422134611644](https://tva1.sinaimg.cn/large/008i3skNgy1gptkwnihlaj30mi0fb0w0.jpg)

### 큐가 가득 찬 경우

더 이상 원소를 넣을 수 없다. 즉, 큐의 길이를 기억하고 있어야 한다.



## 환형 큐의 추상적 자료구조 구현

### 연산의 정의

- size() : 현재 큐에 들어 있는 데이터 원소의 수를 구함
- isEmpty() : 현재 큐가 비어 있는지를 판단
- **isFull() : 큐에 데이터 원소가 꼭 차 있는지를 판단**
- qneueue(x) : 데이터 원소 x를 큐에 추가
- dequeue() : 큐의 맨 앞에 저장된 데이터 원소를 제거(또한, 반환)
- peek() : 큐의 맨 앞에 저장된 데이터 원소를 반환(제거하지 않음)



## 배열로 구현한 환형 큐

> front와 rear를 적절히 계산 하여 배열을 환형을 재활용한다.

### Enqueue를 하는 경우

![image](https://media.vlpt.us/images/inyong_pang/post/2317b8e4-1027-4af6-88c3-ebef26b3885f/image.png)

### Dequeue를 하는 경우

![image](https://media.vlpt.us/images/inyong_pang/post/a97b78b8-5590-4d28-a613-a54f4a5e266d/image.png)

A, B가 dequeue된 상황에서 데이터를 추가 할 수 있다. 즉, G를 enqueue할 수 있다. 배열을 돌려서 사용해야 하기 때문에 rear의 위치를 첫번째로 이동시킨후, rear에 데이터를 추가한다.



### 코드

```python
class CircularQueue:
    def __init__(self, n) -> None:
        self.maxCount = n  			# 인자로 주어진 최대 큐 길이 설정
        self.data = [None] * n  # 빈 큐를 활성화
        self.count = 0 					# 데이터 갯수 
        self.front = -1
        self.rear = -1
    # 현재 큐 길이 반환
    def size(self):
        return self.count

    # 큐가 꽉 차 있는가?
    def isFull(self):
        return self.count == self.maxCount

    def enqueue(self, x):
        if self.isFull():
            raise IndexError("Queue is full")

        self.rear =  (self.rear + 1) % self.maxCount
        print("rear: ", self.rear)
        self.data[self.rear] = x
        self.count += 1

    def dequeue(self):
        if self.count is self.maxCount:
            raise IndexError("Queue is full")

        self.front = (self.front +1) % self.maxCount
        print("front: ", self.front)
        x = self.data[self.front]
        self.count -= 1
        return x

    def peek(self):
        if not self.data:
            raise IndexError("Queue is empty")
        return self.data[self.front]
```

