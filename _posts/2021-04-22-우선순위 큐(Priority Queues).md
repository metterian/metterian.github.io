---
layout: post
title: "우선순위 큐(Priority Queues)"
author: "metterian"
tags: 프로그래머스, 자료구조
---
# 우선순위 큐(Priority Queues)

> 큐가 FIFO(First-In-First-Out) 방식을 따르지 않고, 원소들의 **우선순위**에 따라 큐에서 빠져나오는 방식



## 우선 순위 큐의 구현

우선순위 큐를 구현하는 데에는 두 가지의 서로 다른 접근 방법을 생각할 수 있습니다

1. 큐에 원소를 넣을 때 (**enqueue 할 때) 우선순위 순서대로 정렬해 두는 방법**
2. 큐에서 원소를 꺼낼 때 (dequeue 할 때) 우선순위가 가장 높은 것을 선택하는 방법

### 어느 것이 더 유리 할 까?

**우선순위 순서대로 정렬해 두는 방법**이 좀 더 유리하다.

- 큐의 길이에 비례하는 정도의 시간이 소요가 된다.
- 만약 큐에 데이터 원소들이 우선순위가 상관없이 늘어서 있다고 하면 우선순위가 가장 높은 것을 디큐하려면 모든 원소들을 다 살펴보아야한다.

**연결리스트를 이용하는것이 유리** 하다

- 메모리를 차지하는 양을 보면 선형배열이 공간을 덜 차지



```python
from doublylinkedlist import Node, DoublyLinkedList

class PriorityQueue:

    def __init__(self):
        self.queue = DoublyLinkedList()


    def size(self):
        return self.queue.getLength()

    def isEmpty(self):
        return self.size() == 0

    def enqueue(self, x):
        newNode = Node(x)
        curr = self.queue.head
        while curr.next != self.queue.tail and curr.next.data > newNode.data:
            curr = curr.next
        self.queue.insertAfter(curr, newNode)

    def dequeue(self):
        return self.queue.popAt(self.queue.getLength())

    def peek(self):
        return self.queue.getAt(self.queue.getLength()).data


if __name__ == '__main__':
    myQueue = PriorityQueue()
    myQueue.enqueue(12)
    myQueue.enqueue(1)
    myQueue.enqueue(14)
    myQueue.enqueue(7)

    while not myQueue.isEmpty():
        print(myQueue.dequeue())

```





