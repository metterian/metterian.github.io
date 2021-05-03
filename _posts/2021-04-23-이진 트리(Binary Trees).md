---
layout: post
title: "이진 트리(Binary Trees)"
author: "metterian"
tags: 프로그래머스, 자료구조
---
# 이진 트리(Binary Trees)

> 이진 트리는, 트리에 포함되는 모든 노드의 차수가 2 이하인 트리를 말합니다. 즉, 모든 노드는 자식이 없거나 (리프 노드의 경우), 하나만 있거나, 아니면 둘 있는 세 경우 중 하나에 해당합니다.



## 연산의 정의

- size() - 현재 트리에 포함되어 있는 노드의 수를 구함
- depth() - 현재 트리의 깊이 (또는 높이) 를 구함

- **순회**(traversal)



## 이진 트리의 구현

### 노드(Node)

#### Node

- Data
- Left Child
- Right Child

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gptkh4j6u6j30c40cswfy.jpg" alt="image-20210422163506337" style="zoom:50%;" />

이를 코드로 구현 하면 다음과 같다.

```python
class Node:
    def __init__(self, item) -> None:
        self.data = item
        self.left = None
        self.right = None
```





### 트리(Tree)

root만 가리키고 있으면 자식 노드들을 찾을 수 있다.

```python
class BinaryTree:
    def __init__(self, r) -> None:
        self.root = r
```



### size()

이진트리의 재귀적인 특성을 사용해서 `size()`메소드를 쉽게 구할 수 있다. 다음 그림을 통해 전체 이진트리의 size()를 구할 수 있다.

> 전체 이진트리의 size() = left subtree의 size() + right subtree의 size() + **1(자기 자신)**

![image-20210422163928405](https://tva1.sinaimg.cn/large/008i3skNgy1gptkh7wdaqj30kw0duwid.jpg)

```python
class BinaryTree:
    def size(self):
        # 왼쪽 자식 노드가 존재 할때 사이즈를 구하고, 없으면 0 을 반환 한다(종료 조건)
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return l + r + 1
```



### depth()

`depth()` 메소드 또한 재귀적인 방법을 통해 쉽게 구할 수 있다.

>  $\text{전체 이진트리의 depth()} =  \bold{max}\text{(left subtree의 depth()}, \text{right subtree의 depth()}) + 1$

![image-20210422164856036](https://tva1.sinaimg.cn/large/008i3skNgy1gptkh9qb9hj30jw0a9q5a.jpg)

```python
class Node:
    def depth(self):
        l = self.left.depth() if self.left else 0
        r = self.right.depth() if self.right else 0
        return max(l,r) + 1
```





### 순회(Traversal)

#### DFS(depth first traversal)

DFS 순회의 가장 중요한점은 트리는 각 서브트리로 구성되어 있다는 점이다. 현재 방문 노드를 기준으로 Left **subtree**와 right **subtree**로 구분 짓는 다는 점이 가장 중요하다.

##### 중위 순회 (in-order traversal)

가운데 자기 자신을 방문하는 순회를 말한다.

![image](https://media.vlpt.us/images/inyong_pang/post/9cbf188d-3e74-43a0-b822-256f6ae00abf/image.png)



```python
class Node:
    def inorder(self):
        traversal = []
        if self.left:
            traversal += self.left.inorder()
        traversal.append(self.data)
        if self.right:
            traversal += self.data.inorder()

        return traversal
```

```python
class BinaryTree:
    def inorder(self):
        if self.root:
            return self.root.inorder()
        else:
            return []
```

##### 전위 순회 (pre-order traversal)

자기 자신을 먼저 방문 순회를 말한다.

![image-20210422170120985](https://tva1.sinaimg.cn/large/008i3skNgy1gptkhf8e43j30mi0cb0xa.jpg)

##### 후위 순회 (post-order traversal)

왼쪽, 오른쪽을 방문 하고 마지막에 자신을 방문 순회를 말한다.

![image-20210422170258037](https://tva1.sinaimg.cn/large/008i3skNgy1gptkhg7qgxj30op0cuq7l.jpg)

#### BFS(breadth first traveral)

