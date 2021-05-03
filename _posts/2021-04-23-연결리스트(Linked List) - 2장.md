---
layout: post
title: "연결리스트(Linked List) - 2장"
author: "metterian"
tags: 프로그래머스, 자료구조
---
# 연결리스트(Linked List) - 2장

연결리스트의 가장 큰 장점은 **삽입**과 **삭제**가 유연하는 것이 가장 큰 장점이다. 하지만, 항상 맨 앞 부터 노드를 찾아가는 과정이 비효율적이지 않기 때문에 새로운 메소드를 만들어서 이를 개선 합니다.

다음과 같이 메소드를 추가 해봅시다. 이전과는 들이 `pos` 즉, 인덱스를 주고 노드를 구하는 것이 아니라 노드를 주고 노드를 얻어 봅시다.

- `insertAfter(prev, newNode)` 
- `popAfter(prev)`



### 조금 변형된 연결 리스트

맨 앞에 `dummy node`를 추가한 형태로 새로운 연결리스트를 구현

![image-20210421151103145](https://tva1.sinaimg.cn/large/008i3skNgy1gptkjxbb0mj30qt07xmy5.jpg)

```python
class LinkedList:
    def __init__(self):
        self.nodeCount = 0
        self.head = Node(None) # 더미 노드 추가
        self.tail = None
        self.head.next = self.tail
```



## 연산 정의

- 길이 얻어내기
- 리스트 순회 (traversal)
- 특정 원소 참조 (k 번째)
- 원소 삽입 (insertion)
- 원소 삭제 (deletion)
- 두 리스트 합치기 (concatenation)



### 리스트 순회

기존 과는 달리 `curr`에 next가 존재 할 경우 계속 찾아 가능 형식으로 바뀌였다. 왜냐하면 처음 시작이 dummy 노드 이기 때문.

![image-20210421151509116](https://tva1.sinaimg.cn/large/008i3skNgy1gptkjwqwkwj30fy06sq3d.jpg)

```python
def traverse(self) -> list:
    result = []
    curr = self.head
    while curr.next:
        curr = curr.next
        result.append(curr.data)

    return result
```



### $k$ 번재 원소 얻어내기

이전 과는 달리 dummy 노드가 추가 되었기 때문에 `i=0` 부터 시작한다.

```python
def getAt(self, pos) -> Node:
    if pos < 0 or pos > self.nodeCount:
        return None
    i = 0
    curr = self.head
    while i < pos:
        curr = curr.next
        i += 1
    return curr
```



### 원소의 삽입

`def insertAfter(self, **prev**, newNode):`

- prev가 가리키는 node의 다음에
- newNode를 삽입 하고

```python
    def insertAfter(self, prev, newNode) -> bool:
        newNode.next = prev.next
        # prev가 tail 일경우
        if prev.next is None:
            self.tail = newNode

        prev.next = newNode
        self.nodeCount += 1
        return True

    def insertAt(self, pos, newNode) -> bool:
        if pos < 1 or pos > self.nodeCount + 1:
            return False

        # 연결 리스트가 완전비 비어 있는 경우
        if pos != 1 and pos == self.nodeCount + 1:
            prev = self.tail
        else:
            prev = self.getAt(pos - 1)

        return self.insertAfter(prev, newNode)
```





### 원소의 삭제

- prev의 다음 node를 삭제하고
- 그 node의 data를 리턴

#### 주의 사항

1. prev가 마지막 node일 때 `(prev.next == None)`
   - 삭제할 node가 없음
   - `return None`
2. 리스트 맨 끝에 node를 삭제 할때 `(curr.next == None)`
   - tail 조정 필요



```python
def popAfter(self, prev):
    # prev가 마지막 일때
    if prev.next is None:
        self.nodeCount -= 1
        return None


    curr = prev.next
    # 리스트 맨끝의 node를 삭제 할때 -> tail 조정 필요
    if curr.next is None:
        prev.next = None
        self.tail = prev

    prev.next = curr.next
    self.nodeCount -= 1
    return curr.data


def popAt(self, pos):
    if pos < 1 or pos > self.nodeCount:
        raise IndexError
    prev = self.getAt(pos-1)
    return self.popAfter(prev)
```



### 두 리스트의 연결

head의 next를 연결 하는 것이 키포인트! head가 dummy 노드 임으로 next를 연결 한다

![image-20210421152930344](https://tva1.sinaimg.cn/large/008i3skNgy1gptkk0n7clj30ti0dotb2.jpg)

```python
def concat(self, L) -> None:
    self.tail.next = L.head.next
    # tail이 존재하는 경우 만 tail을 연결
    if L.tail:
        self.tail = L.tail
    self.nodeCount += L.nodeCount
```

