---
layout: post
title: "이진 트리의 넓이 우선 순회 (BFS; Breadth First Traversal)"
author: "metterian"
tags: 프로그래머스 자료구조
---
# 이진 트리의 넓이 우선 순회 (BFS: Breadth First Traversal)

> **원칙**
>
> - 수준(Level)이 낮은 노드를 우선으로 방문
> - 같은 수준의 노드를 사이에는 부모 노드의 방순 순서에 따라 방문(왼 > 오)

![image-20210422203246621](https://tva1.sinaimg.cn/large/008i3skNgy1gptki0tdadj30jn0d30vr.jpg)

순회의 결과: A -> B -> C -> D -> E ... J



## 넓이 우선 순회 흐름 설계

![img](https://tva1.sinaimg.cn/large/008i3skNgy1gptki32lnkj30m10aeq3b.jpg)

- 수준이 낮은 노드 부터 방문해서 큐에 enqueue

![img](https://media.vlpt.us/images/inyong_pang/post/316b58e3-fe09-4844-94d3-cccb73a3f325/image.png)

- A를 방문함(A를 dequeue)

![img](https://media.vlpt.us/images/inyong_pang/post/15e8acca-bc4f-4b91-9191-ccf1afa764e9/image.png)

- A의 자식노드 왼쪽 → 오른쪽 으로

![img](https://media.vlpt.us/images/inyong_pang/post/9d4d74c5-88f6-4e5c-a4c6-51d5c1cee283/image.png)

- B를 방문(B를 dequeue)

![img](https://media.vlpt.us/images/inyong_pang/post/0639a7bb-1e2d-4849-acb8-4d4086966ae4/image.png)

- B의 자식노드(E)를 enqueue

![img](https://media.vlpt.us/images/inyong_pang/post/7d79fa8f-6362-4623-a0cf-3e9ae8561ab3/image.png)

- C를 방문(C를 dequeue)

![img](https://media.vlpt.us/images/inyong_pang/post/cf59013e-b368-411f-bc2f-698975082117/image.png)

- C의 자식노드(G)를 enqueue

![img](https://media.vlpt.us/images/inyong_pang/post/2cc573b8-a233-44e1-87b9-562a461c56fe/image.png)

- D를 방문(D를 dequeue)

![img](https://media.vlpt.us/images/inyong_pang/post/8fb81295-3c3e-45bc-b157-5676c8c03036/image.png)

- D의 자식노드(H)를 enqueue

![img](https://media.vlpt.us/images/inyong_pang/post/a011054a-0ca0-442f-8eaf-5d44aab84702/image.png)

- E를 방문(E를 dequeue), E의 자식노드는 없음

![img](https://media.vlpt.us/images/inyong_pang/post/46a180f3-d08d-4352-9562-6d07871e9731/image.png)

- F를 방문(F를 dequeue)

![img](https://media.vlpt.us/images/inyong_pang/post/528db0a3-4420-489e-8688-bdb6f4af9e13/image.png)

- F의 자식노드(I)를 enqueue

![img](https://media.vlpt.us/images/inyong_pang/post/9c69b76b-34c4-490e-94ac-e1fcb61092e0/image.png)

- G를 방문(G를 dequeue), G의 자식노드 없음

![img](https://media.vlpt.us/images/inyong_pang/post/1caf32da-0f2a-4749-a0f4-3eafd47262c6/image.png)

- H를 방문(H를 dequeue), H의 자식노드 없음

![img](https://media.vlpt.us/images/inyong_pang/post/25ad47c9-ab74-44b3-81c4-8214a3d02f88/image.png)

- I를 dequeue(I를 방문), I의 자식노드 없음, 큐가 비어있으면 순회 끝.



## 넓이 우선 순회 알고리즘 구현

1. traversal에는 빈 리스트를 초기화, q에는 빈 큐를 초기화한다.
2. 빈 트리가 아니면, root node를 큐에 추가(enqueue)한다
3. q가 비어있지 않는 동안 
   1. q의 원소를 추출하여(dequeue) node에 저장한다
   2. node를 방문 처리(traversal에 append)한다
   3. node의 왼쪽, 오른쪽 자식 노드가 있으면 q에 추가한다
4. q가 비어있는 큐가 되면 모든 노드 방문 완료하였음으로 traversal를 리턴한다