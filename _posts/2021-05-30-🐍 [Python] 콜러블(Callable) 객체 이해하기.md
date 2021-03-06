---
layout: post
title: "π [Python] αα©α―αα₯αα³α―(Callable) αα’α¨αα¦ αα΅αα’αα‘αα΅"
author: "metterian"
tags: Python
---
## κ°μ²΄μ μ μ

> κ°μ²΄λ νμ΄μ¬μ΄ λ°μ΄ν°λ₯Ό μΆμνν κ²μλλ€. νμ΄μ¬ νλ‘κ·Έλ¨μ λͺ¨λ  λ°μ΄ν°λ κ°μ²΄λ κ°μ²΄ κ°μ κ΄κ³λ‘ ννλ©λλ€. ν° λΈμ΄λ§(Von Neumann)μ βνλ‘κ·Έλ¨ λ΄μ₯μ μ»΄ν¨ν°(stored program computer)β λͺ¨λΈμ λ°λ₯΄κ³ , λ κ·Έ κ΄μ μμ μ½λ μ­μ κ°μ²΄λ‘ ννλ©λλ€.

<br>

## μ½λ¬λΈ(Callabe) κ°μ²΄

μ°λ¦¬κ° νμ΄μ¬ ν¨μλ ν΄λμ€μμ `()` λ₯Ό μ¬μ©νλ μλ―Έλ λͺ¨λ  μ½λ¬λΈ(Callable) κ°μ²΄λ₯Ό μλ―Έν©λλ€. νμ΄μ¬μμ λͺ¨λ  κ°μ²΄κ° `()` μ¬μ©ν  μ μλ μ΄μ κ° λ°λ‘ μ΄κ² λλ¬Έμλλ€. μ½λ¬λΈ κ°μ²΄λ§μ΄ `()` λ₯Ό μ¬μ©ν  μ μκ³ , μ΄λ `__call__` νΉμ μ΄νΈλ¦¬λ·°ν°(λ©μλ)λ₯Ό μ¬μ©νλ κ²κ³Ό μλ―Έκ° κ°μ΅λλ€. 

```python
>>> class c(object):
...     def f(self): pass
>>> '__call__' in dir(c)
False
>>> '__call__' in dir(c().f)
True
```

<br>

κΈ°λ³Έμ μΌλ‘,`c` ν΄λμ€ μμ μλ `f` λΌλ λ©μλλ μ½λ¬λΈ κ°μ²΄μ΄κΈ° λλ¬Έμ `__call__` λ©μλλ₯Ό ν¬ν¨νκ³  μλ κ²μλλ€. νμ§λ§ `c` λΌλ ν΄λμ€μλ `__call__` μ΄λΌλ λ©μλκ° μ‘΄μ¬νμ§ μμ§λ§ μ½λ¬λΈ κ°μ²΄μλλ€. μλνλ©΄, ν΄λμ€ μμ `__call__`  λ©μλλ₯Ό μ μΈν  μ μκΈ° λλ¬Έμ΄μ£ .

```python
>>> class c(object):
...     def f(self): 
...  				pass
...     def __call__(self):
...         print("callable")
...
>>> '__call__' in dir(c)
True
```

<br>

κ·ΈλΌ Python μΈν°νλ¦¬ν°μμ `c` μ λ¨λμΌλ‘ μ¬μ©ν  λμ, `c()` μ κ°μ΄ μ½λ¬λΈ κ°μ²΄λ‘ μ¬μ©νλ©΄ μ΄λ€μ μ΄ λ€λ₯Έ κ±ΈκΉμ? κ·Έκ²μ λ°λ‘ κ°μ²΄μ μ¬μ©μ μ λ¬΄μλλ€. `()` λ₯Ό μ¬μ©νμ§ μκ³  `c` λ₯Ό λ¨λμΌλ‘ μ¬μ©νκ² λλ€λ©΄ ν΄λμ€κ° μ μ₯λ κ°μ²΄ μ£Όμλ₯Ό λ°ν νκ² λ©λλ€. λ°λ©΄, `c()` μ κ°μ΄ μ½λ¬λΈ κ°μ²΄λ‘ μ¬μ©νκ² λλ€λ©΄ ν΄λμ€μ νΈμΆμ΄ μΌμ΄λκ³  νμ΄μ¬μ ν΄λΉ ν΄λμ€ κ°μ²΄μ μ£Όμλ₯Ό μ°Ύμκ° μμ±κ³Ό μλ©Έμ΄ μΌμ΄λκ²λ©λλ€. λλ¬Έμ, ν΄λμ€λ₯Ό νΈμΆ ν  λ λ§λ€ μλ‘μ΄ μ£Όμκ° κ°μ²΄λ‘ λ°νλλ κ²μ΄μ£ .

```python
>>> class c(object):
...     def f(self): pass
>>> t = C()
>>> t
<__main__.C object at 0x10c932940> # μ£Όμλ₯Ό κ°μ²΄λ‘ λ°ν
>>> t = C()
>>> t
<__main__.C object at 0x10ca152e0> # ν΄λμ€λ₯Ό νΈμΆ ν λλ§λ€ μλ‘μ΄ μ£Όμλ₯Ό κ°μ²΄μ λ°ν
>>> C
<class '__main__.C'>
```



<br>

## ν΄λμ€μμ `()` μ μλ―Έ

ν΄λμ€λ₯Ό νΈμΆ ν  λ μΌλ°μ μΌλ‘ `insatnce = C()` μ κ°μ΄ `()` λ₯Ό μ¬μ©ν΄μ μΈμ€ν΄μ€λ₯Ό ν λΉν©λλ€. μ¦, ν΄λμ€λ₯Ό νΈμΆ νλ κ²μ΄μ§μ(=μ½λ¬λΈ κ°μ²΄λ₯Ό νΈμΆ). κ·Έλ λ€λ©΄ ν΄λμ€λ ν¨μμΈ κ²μΌκΉμ? μ ννλ ν¨μλ νΌμ€νΈ ν΄λμ€λ‘ ν¨μκ° ν΄λμ€μ μΌμ’μλλ€. κ·Έλ λ€λ©΄ ν΄λμ€λ₯Ό νΈμΆμ μ­ν μ λ¬΄μμΌκΉμ?

### ν΄λμ€ μμ±μ

**νμ΄μ¬**μμμ **ν΄λμ€ μμ±μ**μ **μλ©Έμ**μ κ΄ν΄ μμλ³΄κ² μ΅λλ€. **μμ±μ**λ μ΄λ¦μμ μ μ μλ―μ΄ κ°μ²΄κ° λ§λ€μ΄μ§ λ νΈμΆλλ ν¨μλ₯Ό **μμ±μ**λΌκ³  μ΄μΌκΈ° νλ©°, κ°μ²΄κ° μ¬λΌμ§ λ νΈμΆλλ ν¨μλ₯Ό μλ©ΈμλΌκ³  μ΄μΌκΈ°ν©λλ€.

μ°λ¦¬κ° ν΄λμ€λ₯Ό μ¬μ©ν  λ νν μ¬μ©νλ `__init__` μ΄ λ°λ‘ μμ±μ μλκ²μ΄μ§μ. μ°λ¦¬κ° ν΄λμ€λ₯Ό `()` λ₯Ό μ¬μ©ν΄μ νΈμΆνκ² λλ©΄ νμ΄μ¬μ μλμΌλ‘ μ μΌ λ¨Όμ  `__init__`μ μ°Ύμμ¬ μ΄ λ©μλλ₯Ό μ€νμν€λ κ²μ΄μ§μ. μ΄μ°λ λ³΄λ©΄ ν¨μμ μ­νκ³Όλ λΉμ·ν©λλ€. 

### κ° λ©μλ κΈ°λ₯λ€ μ΄ν΄ λ³΄κΈ°

λ€μκ³Ό κ°μ΄ ν΄λμ€λ₯Ό μμ±νκ³  κ° λ©μλμ κΈ°λ₯λ€μ μ΄ν΄ λ³΄κ² μ΅λλ€.

```python
>>> class C:
...     def __init__(self): # μμ±μ
...         print('__init__ method')
...     def __call__(self): # μ½λ¬λΈ κ°μ²΄ μ€μ 
...         print('__call__ method')
...     def special_method(self): # λ©μλ
...         print('special method')
...     def __del__(self): # μλ©Έμ
...         print('__del__ method')
```

<br>

ν΄λμ€λ₯Ό νΈμΆνκ² λλ©΄ κ°μ₯ λ¨Όμ  `__init__` λ©μλκ° κ°μ₯ λ¨Όμ  μ€νλκ³  μ΄ν, μ€νλ κ°μ²΄μ μ£Όμκ° λ°ν λ©λλ€.

```python
>>> C() # ν΄λμ€ νΈμΆ
__init__ method
<__main__.C object at 0x1062208b0>
```

<br>

`del` μ μΈμ€ν΄μ€λ₯Ό μ­μ νλ νμ΄μ¬ λ΄μ₯ν¨μ μλλ€. ν λΉλ μΈμ€ν΄μ€κ° μ‘΄μ¬νμ§ μμΌλ μ€λ₯λ₯Ό λνλ΄λ κ²μλλ€. μ¬κΈ°μ μ€λ₯ λ©μΈμ§λ₯Ό μ΄ν΄ λ³΄λ©΄ *"cannot delete function call"* μ νμΈ ν  μ μμ΅λλ€. μμμ μ€λͺν ν΄λμ€μ νΈμΆμ΄ ν¨μμ νΈμΆκ³Ό λΉμ·ν μ±κ²©μ΄λΌκ³  νμλκ²κ³Ό μΌμΉνλ λΆλΆμΈκ²μ νμΈ ν  μ μμ΅λλ€. 

```python
>>> del C()
  File "<input>", line 1
    del C()
        ^
SyntaxError: cannot delete function call
```

<br>

κ·Έλ λ€λ©΄ νΈμΆμ νΈμΆλ κ°λ₯ ν κΉμ? μμμ `__call__` λ©μλλ₯Ό μ¬μ©νμ¬ μ½λ¬λΈ κ°μ²΄λ₯Ό μ μ ν΄μ€¬κΈ° λλ¬Έμ κ°λ₯ ν©λλ€. μ¬κΈ°μ `__del__` λ©μλκ° μ€νλ κ²μ νμ΄μ¬μμ κ°μ²΄κ° μΈμ€ν΄μ€μ ν λΉ λμ§ μμμ§ λλ¬Έμ λ©λͺ¨λ¦¬ λμλ₯Ό λ°©μ§νκ³ μ μλμΌλ‘ ν λΉλμ§ λͺ»ν κ°μ²΄λ₯Ό μ κ±° νκ²μλλ€. 

```python
>>> C()()
__init__ method
__call__ method
__del__ method
```

<br>

`t` λ³μμ μΈμ€ν΄μ€λ₯Ό ν λΉνλ©΄ λ€μκ³Ό κ°μ΅λλ€. `__call__` μ μ¬μ©ν΄μ μ½λ¬λΈ κ°μ²΄λ‘ μ μ ν΄μ€¬κΈ° λλ¬Έμ `t`  μΈμ€ν΄μ€λ μ½λ¬λΈ μΈμ€ν΄μ€λ‘ μ¬μ©νκ² λκ² μλλ€. μ΄ν, `del` λ΄μ₯ν¨μλ₯Ό μ¬μ©ν΄μ μΈμ€ν΄μ€λ₯Ό μ κ±° ν  μ μμ΅λλ€. 

```python
>>> t = C()
__init__ method
>>> t.special_method()
special method
>>> t() # μ½λ¬λΈ μΈμ€ν΄μ€
__call__ method
>>> del t
__del__ method
```

<br>

## ν¨μμ `()`μ ν΄λμ€μ `()` μ λΉκ΅

κ·Έλ λ€λ©΄ λ©μλ(ν¨μ)μμμ `()` μ¬μ©κ³Ό ν΄λμ€μμμ `()` μ¬μ©μ μ΄λ ν μ°¨μ΄μ μ΄ μμκΉμ?

<br>

### νΌμ€νΈ ν΄λμ€ ν¨μ (First-class function)

>  νΌμ€νΈν΄λμ€ ν¨μλ νλ‘κ·Έλλ° μΈμ΄κ° ν¨μ (function) λ₯Ό **first-class citizen**μΌλ‘ μ·¨κΈνλ κ²μ λ»ν©λλ€. μ½κ² μ€λͺνμλ©΄ ν¨μ μμ²΄λ₯Ό μΈμ (argument) λ‘μ¨ λ€λ₯Έ ν¨μμ μ λ¬νκ±°λ λ€λ₯Έ ν¨μμ κ²°κ³Όκ°μΌλ‘ λ¦¬ν΄ ν μλ μκ³ , ν¨μλ₯Ό λ³μμ ν λΉνκ±°λ λ°μ΄ν° κ΅¬μ‘°μμ μ μ₯ν  μ μλ ν¨μλ₯Ό λ»ν©λλ€.

μ μ μμμ μ€μνκ² μ΄ν΄λ΄μΌν  μ λ³΄λ **first-class citizen** λΌλ κ²μλλ€. νΌμ€νΈ ν΄λμ€ μν°μ¦μ΄λΌλ κ²μ΄ λ¬΄μμ μλ―Ένλ κ²μΌκΉμ? 

μμμ νμ΄μ¬ νλ‘κ·Έλλ°μ ν° λΈμ΄λ§μ΄ μ€κ³ν κ°μ²΄μ κ°λμ μ¬μ©νμ¬ μ΄μλ©λλ€. νλ‘κ·Έλλ°μ ν  λ λλΆλΆμ κΈ°λ₯(ν¨μ, ν΄λμ€..)λ€μ κ°μ²΄ λ¨μλ‘ κ΅¬μ±λκ³  μ€νλμ΄μΌ νλ€λ λ»μ΄μ§μ. νμ§λ§ λ¨μν λ§μ κΈ°λ₯μ κ΅¬ννκΈ° μν΄μ κ°μ²΄ λ¨μλ‘ νλ‘κ·Έλλ°μ νλ©΄ κ΅μ₯ν λΉν¨μ¨μ μΌ κ²μλλ€. 

μλ₯Ό λ€μ΄ λ€μκ³Ό κ°μ΄ λ§μ ν¨μλ₯Ό μ μ νλ€κ³  κ°μ ν΄ λ΄λλ€. λ€μκ³Ό κ°μ΄ λ¨μνκ² ν¨μλ₯Ό μ μΈνμ¬ μ¬μ©ν  μ μμ΅λλ€.

```python
>>> def add(a,b):
...     return a+b
...
>>> add(2,3)
5
```

<br>

νμ§λ§, κ°μ²΄λ₯Ό μ¬μ©ν΄μ μ΄ ν¨μλ₯Ό μ μ νλ€λ©΄ μ΄λ»κ² ν΄μΌ ν κΉμ? μ°λ¦¬κ° μμμ νλ κ°μ²΄λ₯Ό μ½λ¬λΈ κ°μ²΄λ‘ λ§λ€μ΄μ μμ±ν΄μΌ ν©λλ€.

```python
>>> class C(object):
...     def __call__(self, a, b):
...         return a + b
...
>>> t = C() # μΈμ€ν΄μ€ μμ±
>>> add = C() # μΈμ€ν΄μ€ νΈμΆ
>>> add(2,3)
5
```

<br>

νμ΄μ¬μμλ μ΄λ¬ν κ°μ²΄ μ§ν₯μ  νλ‘κ·Έλλ°μ νΌνκ³ μ ν¨μμ κΈ°λ₯(μΈμ, λ°ν)μ λ―Έλ¦¬ νλμ κ°μ²΄λ‘ μ€μ ν΄ λμ΄ λ³΅μ‘ν νλ‘κ·Έλλ°μ λ°©μ§ νκ² μλλ€. 



<br>

## λΆλ‘

### μΈμ€ν΄μ€ λ©μλ(Instance methods)

μΈμ€ν΄μ€ λ©μλλ ν΄λμ€, ν΄λμ€ μΈμ€ν΄μ€μ λͺ¨λ  μ½λ¬λΈ κ°μ²΄ (λ³΄ν΅ μ¬μ©μ μ μ ν¨μ)μ κ²°ν©ν©λλ€.  μ¬κΈ°μ **μΈμ€ν΄μ€μ μλ―Έ**λ κ°μ²΄ μ§ν₯ νλ‘κ·Έλλ°(OOP)μμ ν΄λΉ ν΄λμ€μ κ΅¬μ‘°λ‘ μ»΄ν¨ν° μ μ₯κ³΅κ°μμ ν λΉλ μ€μ²΄λ₯Ό μλ―Έν©λλ€.

μΈμ€ν΄μ€ λ©μλ κ°μ²΄κ° νΈμΆλ  λ, κΈ°λ°μ λλ ν¨μ (`__func__`) κ° νΈμΆλλλ°, μΈμ λͺ©λ‘μ μμ ν΄λμ€ μΈμ€ν΄μ€ (`__self__`) C.κ° μ½μλ©λλ€. μλ₯Ό λ€μ΄, `C` κ° ν¨μ `f()` μ μ μλ₯Ό ν¬ν¨νλ ν΄λμ€μ΄κ³ , `x` κ° `C` μ μΈμ€ν΄μ€μΌ λ, `x.f(1)` λ₯Ό νΈμΆνλ κ²μ `C.f(x, 1)` μ νΈμΆνλ κ²κ³Ό κ°μ΅λλ€. 

μ¬κΈ°μ, **self**λ₯Ό λΆμΈ μͺ½μ **bound**, μ λΆμΈ μͺ½μ unbound λ©μλλΌ ν©λλ€.

#### μμ 

```python
>>> class C:
...     def method_one():
...         print("method one called")
...     def method_two(self):
...         print("method two called")
...     @staticmethod # μ μ  λ©μλ
...     def method_three():
...         print("method three called")
```

μμ μ½λλ₯Ό μμ  μ½λλ‘ μ¬μ©νκ³ μ ν©λλ€. `method_three()` λ λ°μ½λ μ΄ν°λ₯Ό μ¬μ©νμ¬ μ μ  λ©μλλ‘ μ¬μ©ν©λλ€. μ΄ λ©μλλ₯Ό μ¬μ©νλ©΄ `method_three()` λ₯Ό bound methodλ‘ λ§λ€μ§ λ§λΌκ³  μ€μ  ν  μ μμ΅λλ€. 

`method_one()` μ μ€νν κ²°κ³Όλ λ€μκ³Ό κ°μ΅λλ€.

```python
>>> C.method_one
<function C.method_one at 0x10ca07b80>  # λ¨μ κ°μ²΄μ μ£Όμ λ°ν
>>> C.method_one()
method one called # Cμ μΈμ€ν΄μ€κ° μ‘΄μ¬νμ§ μκΈ° λλ¬Έμ μ€ν κ°λ₯
>>> C.method_one(t)
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    C.method_one(t)
TypeError: method_one() takes 0 positional arguments but 1 was given
```

<br>

λ€μμΌλ‘ 

`method_two(self)` λ₯Ό μ€νν κ²°κ³Όλ λ€μκ³Ό κ°μ΅λλ€.

```python
>>> C.method_two(t) # t = C(); t.method_two()μ κ°μ μλ―Έ
method two called
>>> C.method_two()
Traceback (most recent call last):
  File "<input>", line 1, in <module>
    C.method_two()
TypeError: method_two() missing 1 required positional argument: 'self'
```

μ¬κΈ°μ λμ¬κ²¨ λ΄μΌν  μ½λλ `C.method_two(t)` μλλ€. μ°λ¦¬λ μ΄μ μ ν΄λμ€μ μΈμ€ν΄μ€λ₯Ό ν λΉν λ€μ μΈμ€ν΄μ€μ λ©μλλ₯Ό νΈμΆ νμμ΅λλ€. ν΄λμ€ λ΄λΆμμ `self` λΌλ μΈμλ₯Ό μ μΈ νμ§λ§ ν΄λΉ λ©μλλ₯Ό μ¬μ©ν  λ, μΈμλ‘ κ°μ λ£μ΄μ£Όμ§ μμλ μ€νμ΄ κ°λ νμ΅λλ€. μλνλ©΄ νμ΄μ¬μμλ μΈμ€ν΄μ€μ λ©μλκ° νΈμΆ(call)λλ©΄ μλμΌλ‘ μΈμ€ν΄μ€ κ°μ²΄λ₯Ό ν΄λΉ λ©μλμ μΈμ μ¦ `self`λ‘ λκ²¨μ£Όμλ κ²μλλ€. 

κ·Έλ¬λ―λ‘ `C.method_two(t)`μ μλ―Έλ μΈμ€ν΄μ€λ₯Ό λ³μμ ν  λΉνμ§ μκ³ , λͺμμ μΌλ‘ `self` λΌλ λ³μμ μΈμ€ν΄μ€λ‘ μ¬μ©ν  λ³μλ₯Ό μ§μ΄ λ£μ΄ μ΄λ₯Ό μ€ν ν κ²μ΄μ§μ.

```python
t = C()
t.method_two()
```

