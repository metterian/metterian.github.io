Streamlit은 파이썬으로 웹 애플리케이션을 쉽게 만들 수 있도록 도와주는 라이브러리입니다. Streamlit에서 제공하는 캐싱 기능을 사용하면 애플리케이션의 속도를 향상시킬 수 있습니다. 이번에는 Streamlit의 새로운 캐싱 기능인 `@st.experimental_memo`와 @st.experimental_singleton에 대해 알아보겠습니다.

`@st.cache`는 데이터를 캐싱하고 global object(TensorFlow sessions, database connections, etc)를 저장하기 위한 용도로 사용됩니다. 하지만 `@st.cache`는 느리고 메모리를 많이 사용하는 단점이 있습니다. 이를 보완하기 위해 새로운 캐싱 기능인 `@st.experimental_memo`와 @st.experimental_singleton이 도입되었습니다. 이 기능들은 `@st.cache`와 비교하여 10배 이상 빠르게 동작합니다.

`@st.experimental_memo`는 기본적으로 `@st.cache`를 대체합니다. `@st.experimental_memo`는 데이터를 저장하는 용도로 사용됩니다. 데이터를 여러 번 실행하고 싶지 않은 값비싼 계산을 캐싱하는 데 사용됩니다. 예를 들어, 데이터프레임 계산, 데이터 다운로드, pi값 계산 등이 이에 해당됩니다.

`@st.experimental_singleton`은 non-data objects를 저장하기 위한 용도로 사용됩니다. 계산 결과가 아니라 대신 계산 또는 기타 프로그램 논리를 구현하는 데 사용되는 개체가 있는 경우에 사용됩니다. 예를 들어, TensorFlow 세션, 데이터베이스 연결 등이 이에 해당됩니다.

이제 `@st.experimental_memo`와 `@st.experimental_singleton`을 어떤 경우에 사용해야 하는지 알아보겠습니다.

- 데이터프레임 계산 (pandas, numpy, 등): memo
- 다운로드한 데이터 저장: memo
- pi값 계산: memo
- TensorFlow 세션: singleton
- 데이터베이스 연결: singleton


아래는 `@st.experimental_singleton`를 사용하는 예시 코드입니다.



```python
Copy code
import streamlit as st
from transformers import BertModel

@st.experimental_singleton
def get_database_session(url):
     # Create a database session object that points to the URL.
     return session

@st.experimental_singleton
def get_model(model_type):
    # Create a model of the specified type.
    return BertModel.from_pretrained(model_type)

if st.button("Clear All"):
    # Clears all singleton caches:
    st.experimental_singleton.clear()

```



위의 코드에서 `get_database_session()`과 `get_model()` 함수는 `@st.experimental_singleton`을 사용하여 정의되었습니다. 이 함수들은 각각 데이터 베이스 연결과 모델을 가져옵니다. 이 함수들은 데이터베이스 연결과 모델을 생성하고, 캐싱하여 불필요한 계산을 피합니다. `@st.experimental_singleton` 데코레이터는 이전 결과를 캐싱하고, 이전과 동일한 인수가 함수에 전달될 때 이전 결과를 반환합니다.

마지막으로, `if st.button("Clear All"):` 코드는 `@st.experimental_singleton` 캐시를 모두 지우는 기능을 제공합니다.

`@st.experimental_memo`와 `@st.experimental_singleton`은 각각 데이터 저장과 데이터 저장이 아닐 때 사용되므로, 적절히 사용하여 애플리케이션의 성능을 향상시킬 수 있습니다.
