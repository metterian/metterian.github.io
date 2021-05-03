---
layout: post
title: "연결 리스트(Linked Lists) - 1장"
author: "metterian"
tags: 프로그래머스, 자료구조
---
#  연결 리스트(Linked Lists) - 1장

> 선형 배열이 "번호가 붙여진 칸에 원소들을 채워넣는" 방식이라고 한다면, 연결 리스트는 "각 원소들을 줄줄이 엮어서" 관리하는 방식입니다.



## Table of Contents

[toc]





## Node

- Data
- Link (next)

Node 내의 데이터는 다른 구조로 이루어질 수 있다.
ex) 문자열, 레코드. 또 다른 연결 리스트 등

## 기본적인 연결 리스트

기본적인 연결 리스트는 **Head**, **Tail**, **number of nodes** 로 구성된다. 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gptkk6bsw3j30z40dxgp2.jpg" alt="image-20210420234742311" width="300" />

## 자료 구조 정의

### Node

node는 data와 link(next)로 구성된다. 이를 Python 클래스로 구현하면 다음과 같다.

```python
class Node:
    def __init__(self, item) -> None:
        self.data = item # item을 받아서 data에 저장
        self.next = None
```

이를 그림으로 표현 하면 다음과 같다.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gptkk89q40j30f70b675l.jpg" alt="image-20210420235129159" width="300" />

### Linked List

처음에 연결리스트를 생성할 때 비어있는 연결리스트를 생성하기 위해서 다음과 같이 초기화를 해준다. 

```python
class LinkedList:
    def __init__(self) -> None:
        self.nodeCount = 0
        self.head = None
        self.tail = None
```

이를 그림으로 표현 하면 다음과 같다.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gptkka4p70j30hj0dzgnx.jpg" alt="image-20210420235345923" width= "300" />

## 연산 정의

1. 특정 원소 참조 ($k$ 번째)
2. 리스트 순회
3. 길이 얻어내기
4. 원소 삽입
5. 원소 삭제
6. 두 리스트 합치기



### 1. 특정 원소 참조

특정 원소를 찾아가는 코드를 구현하면 다음과 같다.

```python
def getAt(self, pos):
    if pos <= 0 or pos > self.nodeCount: # 찾는 위치가 불가능 할 경우
        return None

    i = 1
    curr = self.head
    while i < pos:
        curr = curr.next
        i += 1

    return curr
```



### 2. 원소 탐색

```python
def traverse(self):
    answer = []
    curr = self.head
    while curr:
        answer.append(curr.data)
        curr = curr.next

    print(answer)
    return answer
```



#### Array와 Linked List 비교

|                | 배열        | 연결 리스트      |
| -------------- | ----------- | ---------------- |
| 저장 공간      | 연속한 위치 | 임의의 위치      |
| 특정 원소 지칭 | 매우 간편   | 선형 탐색과 유사 |
| 시간 복잡도    | $O(1)$      | $O(n)$           |



### 3. 원소의 삽입

`def insertAt(self, pos, newNode):` 는 다음과 같이 구현 된다.

- pos가 가리키는 위치에 (` 1 ≤ pos ≤ nodeCount+1)
- newNode를 삽입하고
- 성공/실패에 따라 True/False를 반환

그림으로 표현하면 다음과 같다. `L.insertAt(pos, newNode)`

![image-20210421002443785](https://tva1.sinaimg.cn/large/008i3skNgy1gptkkdak6oj30ve0bndh4.jpg)

처음의 연결리스트는 위와 같이 표현되고, 삽입 진행시 다음과 같이 표현된다.

![image-20210421002810897](https://tva1.sinaimg.cn/large/008i3skNgy1gptkkhnepqj30x70g878k.jpg)

위와 같이 연결리스트를 구현하기 위해서 먼저 `pos-1` 노드를 찾아야 한다. 이를 `prev` 변수로 지정한다. 이후 다음과 같은 순서로 `newNode` 를 삽입하게 된다.

1. `newNode` 를 `pos(=prev.next)` 에 연결한다.
2. `prev` 의 next를 `newNode`에 연결한다.
3. `nodeCount += 1`을 진행한다. 

#### 코드 구현 주의 사항

1. 삽입하려는 위치가 리스트 맨 앞일때
   - prev가 존재 하지 않은다.
   - Head 조정 필요
2. 삽입하려는 위치가 리스트 맨 끝일때
   - Tail 조정 필요

이를 코드로 구현하면 다음과 같다.

```python
def insert(self, pos, newNode) -> bool:
    if pos < 1 or pos > self.nodeCount + 1:
        return False

    if pos == 1: # 맨 앞일 경우
        newNode.next = self.head
        self.head = newNode
    else:
        if pos == self.nodeCount + 1: # 앞부터 찾지 말고 맨 뒤부터 할당
            prev = self.tail
        else:
            prev = self.getAt(pos - 1) # pos -1 노드를 찾는다.
        newNode.next = prev.next
        prev.next = newNode

    if pos == self.nodeCount + 1: #맨 뒤일 경우
        self.tail = newNode
    self.nodeCount += 1
    return True
```



#### 연결리스트 원소 삽입 복잡도

- 맨 앞에 삽입하는 경우: O(1)
- 중간에 삽입하는 경우: O(n)
- 맨 뒤에 삽입하는 경우: O(1)



### 4. 원소의 삭제

- pos가 가리키는 위치의 (1 ≤ pos ≤ nodeCount)
- node를 삭제하고
- 그 node의 데이터를 리턴

이를 그림으로 설명하면 다음과 같다.

![image-20210421125829455](https://tva1.sinaimg.cn/large/008i3skNgy1gptkkfnkjpj30q90d740l.jpg)

`prev` 가 가리키는 대상을 curr가 가리키는 대상으로 치환 시켜 준다. 그리고, `nodeCount` 를 1만큼 감소 시켜준다.

#### 코드 구현 주의 사항

##### 삭제하려는 node가 맨 앞의 것일 때

- prev 가 없음
- Head 조정 필요

##### 리스트 맨 끝의 node를 삭제 할 때

- Tail 조정 필요

##### 삭제하려는 node가 마지막 node 일 때

이는 즉, `pos == nodeCount` 인 경우? 를 묻는 말과 같다.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gptkkgnjt4j30hg0cedgu.jpg" alt="image-20210421130159437" width="300" />

이런 경우 한번에 처리 할 수 없다. (prev를 찾을 방법이 없기 때문) 즉, 앞에서 부터 찾아 와야 한다. 이를 코드로 구현 하면 다음과 같다.

```python
    def popAt(self, pos):
        if pos < 1 or pos > self.nodeCount:
            raise IndexError
        
        curr = self.getAt(pos) # 삭제 할 노드
        if pos == 1: # head 노드를 삭제 할 경우
            if self.nodeCount == 1: # 노드 갯수가 한 개 일 때
                self.head = None
                self.tail = None
                self.nodeCount = 0
            else:
                self.head = self.head.next
                self.nodeCount -= 1
            return curr.data
        
        else: 
            prev = self.getAt(pos - 1)
            if pos == self.nodeCount: # tail 노드를 삭제 할 경우
                prev.next = None
                self.tail = prev
                
            else:
                prev.next = curr.next
        self.nodeCount -= 1
        return curr.data
```



#### 연결 리스트 원소 삭제 복잡도

- 맨 앞에 삽입하는 경우: $O(1)$
- 중간에 삽입하는 경우: $O(n)$
- 맨 뒤에 삽입하는 경우: $O(n)$



### 5. 두 리스트의 연결 연산

```python
def concat(self, L):
    self.tail.next = L.head
    # tail이 존재하는 경우 만 tail을 연결
    if L.tail:
        self.tail = L.tail
    self.nodeCount += L.nodeCount
```



