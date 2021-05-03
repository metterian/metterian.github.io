---
layout: post
title: "양방향 연결 리스트 (Doubly Linked Lists)"
author: "metterian"
tags: 프로그래머스, 자료구조
---
# 양방향 연결 리스트 (Doubly Linked Lists)

> 한 쪽으로만 링크를 연결하지 말고, 양쪽으로! -> 앞으로도 (다음 node) 뒤로도 (이전 node) 진행이 가능 해진다. 양방향 연결 리스트가 단방향 연결 리스트에 비해서 가지는 장점은, 데이터 원소들을 차례로 방문할 때, 앞에서부터 뒤로도 할 수 있지만 뒤에서부터 앞으로도 할 수 있다는 점입니다. 

![image-20210421154758712](https://tva1.sinaimg.cn/large/008i3skNgy1gptkj0mfxxj30r103yab3.jpg)



## Table of Contents

[toc]





### Node 구조의 확장

기존 연결 리스트를 확장해서 사용해야 한다. 그림으로 표현 하면 다음과 같다.

![image-20210421154940507](https://tva1.sinaimg.cn/large/008i3skNgy1gptkj1x99uj30c904h3yr.jpg)

```python
class Node:
    def __init__(self, item):
        self.data = item
        self.prev = None # 이전 노드가 추가됨
        self.next = None
```

#### 리스트와 처음과 끝에 dummy node를 두자!

이를 통해 데이터를 담고 있는 node들은 모두 같은 모양이 된다.

![image-20210421155231990](https://tva1.sinaimg.cn/large/008i3skNgy1gptkj46fggj30rv08zmye.jpg)

```python
class DoublyLinkedList:
    def __init__(self):
        self.nodeCount = 0
        self.head = Node(None)
        self.tail = Node(None)

        self.head.prev = None
        self.head.next = self.tail

        self.tail.prev = self.head
        self.tail.next = None
```

위 코드를 그림으로 표현 하면 다음과 같다.

![image-20210421155439347](https://tva1.sinaimg.cn/large/008i3skNgy1gptkj79ektj30fz07ogm6.jpg)



### 리스트 순회

tail 에도 dummy 노드가 포함 되어 있기 때문에 `curr.next.next`가 유효 할 때 가지로 조건식이 변경 되었다.

![image-20210421155515670](https://tva1.sinaimg.cn/large/008i3skNgy1gptkj9rfxuj30ir07s74t.jpg)

```python
def traverse(self) -> list:
    result = []
    curr = self.head
    while curr.next.next:
        curr = curr.next
        result.append(curr.data)

    return result
```



### 리스트 역순회

양방향 연결을 통해 리스트 역순회도 가능해 진다.

![image-20210421155707211](https://tva1.sinaimg.cn/large/008i3skNgy1gptkjdzvguj30ih07y3z1.jpg)

```python
    def reverse(self) -> list:
        result = []
        curr = self.tail
        while curr.prev.prev:
            curr = curr.prev
            result.append(curr.data)

        return result
```





### 원소의 삽입

`L.insertAfter(prev, newNode)`메소드는 다음과 같은 순서로 구해진다.

먼저, `next= prev.next` 변수를 구한다.

![image-20210421155945314](https://tva1.sinaimg.cn/large/008i3skNgy1gptkjbnd7qj30q10cb75o.jpg)

그런 뒤에, 어떠한 노드도 끈기 전에 newNode에 next와 prev 에 링크를 연결한다.

![image-20210421160029618](https://tva1.sinaimg.cn/large/008i3skNgy1gptkjcz5r7j30sa0c6taz.jpg)

이 과정이 완료되면 prev와 next의 연결을 끊을 준비가 되었다. prev의 next를 newNode로 가리키고, next의 prev를 newNode를 가리키게 한다. 마지막으로, nodeCount에 1을 더 해준다.

![image-20210421160118217](https://tva1.sinaimg.cn/large/008i3skNgy1gptkjjab2yj30ql0c5ac0.jpg)

![image-20210421160125039](https://tva1.sinaimg.cn/large/008i3skNgy1gptkjfeeg3j30qu0bqwgl.jpg)

![image-20210421160233158](https://tva1.sinaimg.cn/large/008i3skNgy1gptkjobrcuj30pp0abwg1.jpg)

이를 코드로 구현하면 다음과 같다.

```python
def insertAfter(self, prev, newNode) -> bool:
    next = prev.next
    newNode.prev = prev
    newNode.next = next

    prev.next = newNode
    next.prev = newNode

    self.nodeCount += 1
    return True
```



### 특정 원소 찾기 

이전 코드를 변형 해서 사용한다. 왼쪽에 있으면 head부터 찾아 가고 오늘쪽에 있으면 tail 부터 찾아 간다. 

```python
    def getAt(self, pos) -> Node:
        if pos < 0 or pos > self.nodeCount:
            return None
        
        if pos > self.nodeCount // 2:
            i = 0
            curr = self.tail
            while i < self.nodeCount - pos + 1:
                curr = curr.prev
                i += 1
        else:
```

