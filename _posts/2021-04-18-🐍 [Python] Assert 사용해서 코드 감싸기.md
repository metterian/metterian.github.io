---
layout: post
title: "🐍 [Python] Assert 사용해서 코드 감싸기"
author: "metterian"
tags: Python
---



## **Background**

이 글은 Python Tricks The Book 이라는 책 2장에 소개되는 파이썬의 Assertion 에 관한 내용을 정리했습니다.

파이썬을 사용한 프로젝트를 진행하거나 코드를 읽다보면 종종 Assertion 구문을 만나곤 합니다. 정확하게 이 구문이 언제 사용되는지 혹은 어떻게 사용해야 하는지 알지 못하고 넘어가곤 했습니다.

읽은 책의 내용을 토대로 Assertion 구문을 **언제** 그리고 **어떻게** 사용하는지 알아보겠습니다.

---

## **Study**

### **Assertion**

Assertion 구문은 어떤 조건을 테스트하는 **디버깅 보조 도구** 라는 것이 핵심입니다.

아래의 코드는 assert 가 사용되는 예시입니다.

온라인 쇼핑몰에서 할인 쿠폰 기능을 시스템에 추가하고, 다음과같은 `apply_discount` 함수를 작성했습니다.

```python
def apply_discount(product, discount):
    price = int(product['price'] * (1.0 - discount))
    assert 0 <= price <= product['price']
    return price
```

의도대로라면, 이 함수로 계산된 가격은 0원보다 낮을 수 없고, 할인되었기 때문에 원래의 가격보다 높으면 안됩니다.

일반적인 경우라면, 할인율(`discount`)이 0이상 1이하의 범위일 것입니다. 이런 경우에는 당연하게도 할인된 가격이 assert 구문의 조건을 참으로 만들게 됩니다.

하지만, 할인율(`discount`)이 0이상 1이하의 범위가 아니라면 어떨까요? 예를들면 할인율이 2가 되면, price는 음수가 될 것입니다. 즉, 상품을 사는 고객에게 돈을 더 줘야 됩니다.

다행히도 assert 구문에서는 이런 경우에 assert 구문의 조건이 거짓이 되므로 `AssertionError`라는 예외가 발생하게 됩니다.

만약 자신이 이 코드를 작성한 프로그래머라고 가정해 보겠습니다. 이 함수내에 Assertion 구문이 없었다면, 쇼핑몰을 운영하는 도중 문제가 발생했을 때, 디버깅 하는것이 생각보다 쉽지 않을 수 있습니다.

반대로, Assertion 구문을 적절하게 위치시켜 버그 상황시에 `AssertionError` 예외가 발생한다면 위치에 대한 스택트레이스(stacktrace)를 확인하여 버그를 쉽게 디버깅 할 수 있을것입니다.

이것이 Assertion 구문이 가지는 힘입니다.

### 🤔 **일반 예외처리와 무엇이 다른가?**

Assertion 구문은 일반적인 `if`구문 `try - except` 구문을 사용한 예외처리와 다른 역할을 합니다. 예를 들면 `File-Not-Found`와 같은 예상되는 에러 조건을 검사하기 위해 사용되는것은 올바른 활용 방식이 아닙니다.

이 구문은 예상하지 않은 프로그램의 상태를 확인하기 위해 활용해야 합니다. 구문의 조건을 만족하지 않으면 프로그램이 정상적으로 실행되지 않고 종료되는데, 이는 프로그램의 버그가 있다는 것을 의미합니다.

이런 특징으로 비추어 볼 때, Assertion 구문이 런타임 환경이 아닌 **디버깅 환경** 에 도움을 주는 역할을 한다는 것을 알 수 있습니다. 개발자는 이를 토대로 개발환경에서 편안하게 디버깅하게 됩니다.

### 📑 **문법**

`assert_stmt ::= "assert" expression1 [",", expression2]`

`expression1`은 테스트 조건이고, 뒤의 `expression2`는 테스트 조건이 거짓일 때, 예외의 메시지로 전달할 메시지입니다.

### 🚨 **주의 사항**

위의 문법을 인터프리터가 해석하는 방식을 간단한 토막코드로 만들게되면 다음과 같습니다.

```python
if __debug__:
    if not expression1:
        raise AssertionError(expression2)
```

이 코드를 보면, 앞서 설명했던 런타임 환경이 아닌 디버깅 환경에 도움을 주는 역할을 해야만 하는 이유를 이해할 수 있습니다.

Assertion 구문은 `__debug__`라는 전역변수를 검사를 합니다. 이 전역변수는 일반적인 상황에서는 항상 참이지만 최적화가 필요한 경우에는 거짓이 되게 됩니다.

따라서, Assertion 구문을 예외처리에 잘못 활용하게된다면, 코드가 의도한대로 동작하지 않을 수 있습니다.

예를들면 데이터 유효성 검증을 하는데 Assertion 구문을 사용하게 된다면 어떨까요?

```python
def delete_product(prod_id, user):
    assert user.is_admin(), 'Must Be Admin'
    assert store.has_product(prod_id), 'Unknown product'
    store.get_product(prod_id).delete()
```

일반적인 경우라면 `__debug__` 전역변수는 참이므로 사용자에 대한 권한 확인과 제품이 존재하는지 확인하는 과정이 올바르게 진행될 것입니다.

반면, `PYTHONOPTIMIZE`와 같은 환경변수 설정으로 인해 Assertion 구문이 비활성화가 된다면 위의 함수는 의도와 동작하지 않게 될 것이고 이는 큰 장애로 귀결될 수 있습니다.

결국, 이런 문제를 회피하기 위해서는 데이터 유효성 검증시에 Assertion 구문을 절대 사용하지 말아야 합니다. 대신 유효성 검사에서는 `if` 구문 등을 사용하여 처리하고 예외를 발생시켜야 합니다.

또한, 절대 실패하지 않는 단언문을 주의해야합니다.

`assert(1 == 2, 'This should fail')`

위의 단언문은 절대 실패하지 않습니다. 왜냐하면, 튜플은 비어있지 않는이상 항상 조건이 참이 되기 때문입니다. 이런 직관적이지 못한 동작 때문에 실수를 하는 경우가 종종 있습니다.

```python
assert (
  counter == 10,
  'It should ave counted all the items'
)
```

위와 같이 여러 줄에 걸쳐서 Assertion 구문을 작성하게 되면, 잘못된 구문이라는 것을 알기가 더 어려워 질 수도 있습니다. 따라서 이 구문을 사용할 때 튜플을 사용하는것에 주의를 해야 합니다.

이런 주의사항들이 있더라도 파이썬의 Assertion 구문은 적재적소에 재대로 사용하기만 한다면 디버깅하는 과정에 많은 도움을 주어 생산성을 늘리는것은 물론 유지 보수가 쉬운 프로그램을 작성할 수 있을것입니다.