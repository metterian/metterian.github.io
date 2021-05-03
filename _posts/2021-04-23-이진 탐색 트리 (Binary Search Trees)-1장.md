---
layout: post
title: "이진 탐색 트리 (Binary Search Trees)-1장"
author: "metterian"
tags: 프로그래머스, 자료구조
---
# 이진 탐색 트리 (Binary Search Trees)

> 모든 노드에 대해서,
>
> - 왼쪽 서브트리에 있는 데이터는 모두 혀냊 노드의 값보다 작고
> - 오른쪽 서브트리에 있는 데이터는 모두 현재 노드의 값보다 큰 
>
> 위 성질을 만족하는 이진 트리 (단, 중복 데이터는 없다고 가정한다)

 이진 탐색을 적용하기 위해서는 탐색 대상인 배열이 미리 정렬되어 있어야 하므로, 이 배열에 데이터 원소를 추가하거나 배열로부터 데이터 원소를 삭제하는 일은 `n` 에 비례하는 시간을 소요합니다.



## Table of Contents

[toc]



## 정렬된 배열을 이용한 이진 탐색과 비교

![img](https://media.vlpt.us/images/inyong_pang/post/7321a994-acd5-4905-91f6-794419f21ee5/image.png)

### 장점

- 이진 탐색 트리를 이용하면, 배열을 이용하는 것 보다
  **데이터의 추가**, **삭제가 용이**하다

### 단점

- 하지만, 공간을 많이 차지한다는 단점
- 시간복잡도 **O(logN)를 갖고 있진 않다**



## 이진 탐색 트리의 추상적 자료구조

![img](https://media.vlpt.us/images/inyong_pang/post/aafd2b60-cbd1-49dd-862c-015a3488ca91/image.png)

- 데이터 표현
- 각 노드는 **key-value 쌍**으로 표현
- 키를 이용해서 검색가능
- 복잡한 데이터 레코드로 확장 가능
- 숫자가 key
- 이름이 value

## 연산의 정의

- insert(key, data) : 트리에 주어진 데이터 원소를 추가
- remove(key) : 특정 원소를 트리로 부터 삭제
- lookup(key) : 특정 원소를 검색
- inorder() : 키의 순서대로 데이터 원소를 나열
- min(), max() : 최소 키, 최대 키를 가지는 원소를 각각 탐색

## 이진 탐색 트리에 원소 삽입

![image](https://media.vlpt.us/images/inyong_pang/post/4feae03f-c3c8-4f51-b939-837bef47a682/image.png)

## 이진 탐색 트리 구현

### 초기화

```python
class Node:
    # 초기화
    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.left = None
        self.right = None
        
class BinSearchTree:
    # 저번에는 인자를 주었는데 이번에는 none으로 초기화
    def __init__(self):
        self.root = None
```

### inorder traversal

```python
class Node:

    # 이번에는 노드들의 리스트를 만들어서 리턴한다.
    def inorder(self):
        traversal = []
        if self.left:
            traversal += self.left.inorder()
        traversal.append(self)
        if self.right:
            traversal += self.right.inorder()
        return traversal

class BinSearchTree:

    def inorder(self):
        if self.root:
            return self.root.inorder()
        else:
            return []
```

### min(), max()

```python
# 노드 클래스
class Node:
    
    def min(self):
        if self.left:
            return self.left.min()
        else:
            return self

    def max(self):
        if self.right:
            return self.right.max()
        else:
            return self
            
            
# 이진 탐색 트리 클래스
class BinSearchTree:
    
    def min(self):
        # 루트 노드가 존재하면
        if self.root:
            return self.root.min()
        else:
            return None

    def max(self):
        if self.root:
            return self.root.max()
        else:
            return None
```



### lookup()

- 입력인자 : 찾으려는 대상 키
- 리턴 : **찾은 노드**, 그것의 **부모 노드**(삭제에 이용됨). 각각 없으면 None으로 리턴

```python
# 노드 클래스
class Node:

    # parent 인자가 주어지지않으면 None으로 생각하라는 말
    def lookup(self, key, parent=None):
        # 지금 방문된 노드(self.key)보다 탐색하려는 키가 작으면 왼쪽으로 가야함
        if key < self.key:
            # 왼쪽 자식이 있을 때
            if self.left:
                # 재귀적으로 호출
                return self.left.lookup(key, self)
            else:
                # 찾으려는 키가 없구나
                return None, None
        
        # 지금 방문된 노드보다 찾으려는 키가 크면 오른쪽으로 가야함
        elif key > self.key:
            # 오른쪽 자식이 있을 때
            if self.right:
                return self.right.lookup(key, self)
            else:
                return None, None
        
        # 찾았다 해당 노드!
        else:
            return self, parent
            
            
# 이진 탐색 트리 클래스
class BinSearchTree:

    def lookup(self, key):
        if self.root:
            return self.root.lookup(key)
        else:
            return None, None
```

### insert()

- 입력인자 : 키, 데이터 원소
- 리턴 : 없음

```python
class Node:
    def insert(self, key, data):
        # 찾으려는 키가 해당노드보다 작은 경우 왼쪽으로
        if key < self.key:
            # 왼쪽 자식 노드가 존재하는 경우
            if self.left:
                self.left.insert(key, data)
            # 존재하지않으면 노드를 단다.
            else:
                self.left = Node(key, data)
                
        # 찾으려는 키가 해당 노드보다 큰 경우 오른쪽으로        
        elif key > self.key:
            # 오른쪽 자식 노드가 존재하는 경우 
            if self.right:
                self.right.insert(key, data)
            # 존재하지 않으면 노드를 단다.
            else:
                self.right = Node(key, data)
                
        # 중복된 노드가 존재하는 경우 에러 발생
        else:
            print("중복된 노드가 존재")

        return True
        
        
class BinSearchTree:
    # 노드 삽입
    def insert(self, key, data):
        # 존재하는 트리라면
        if self.root:
            self.root.insert(key,data)
        
        # 트리가 존재하지않다면 해당 노드를 루트에 넣는다.
        else:
            self.root = Node(key, data)
```