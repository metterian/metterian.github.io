---
layout: post
title:  "๐ฆ How to build a State-of-the-Art Conversational AI"
author: "metterian"
tags: AI NLP
---



# How to build a State-of-the-Art Conversational AI with Transfer Learning

์ด ๊ธ์ **ConvAI2  NeurIPS**(2018) ๋ํ์์ **SOTA**(state-of-the-art)๋ฅผ ๊ธฐ๋กํ **Hugging Face** ์ Conversation AI์ ๋ํ ํํ ๋ฆฌ์ผ๋ฅผ ๋ฒ์ญํ ํฌ์คํธ์๋๋ค. ์กธ์ ํ๋ก์ ํธ๋ฅผ ์งํํ๋ ํ๋ถ์ ์์ค์์ ์์ฑํ ๊ธ์ด๋ ์ฐธ๊ณ ํ๊ณ  ๋ด์ฃผ์๋ฉด ๊ฐ์ฌํ๊ฒ ์ต๋๋ค.๐ 



> ๋ค์ ์ฌ์ดํธ์์ Hugging Face๊ฐ ์ ์ํ ๊ฐ๋จํ ๋ฐ๋ชจ๋ฅผ ์ฒดํ ํด ๋ณด์ค ์ ์์ต๋๋ค.๐ฎ*[convai.huggingface.co](https://convai.huggingface.co/)*. 

![img](https://miro.medium.com/max/600/1*Fn0bcNyyEyqpGq-nCPyoYw.gif)



## ๊ธ์ ์ฃผ์ ๋ชฉํ

- OpenAI์ธ GPT ์ GPT-2 Transformer language ๋ชจ๋ธ์ ์ฌ์ฉํ์ฌ ์ ์ดํ์ต์ ํตํ ์ต์ ๋จ ๋ํ ์์ด์ ํธ๋ฅผ ์ ์
- [ConvAI2](http://convai.io/)์์ ์ฐ์นํ NeurIPS 2018 ๋ํ๋ฅผ ์ฌ์ฉ

> ์ด ํํ ๋ฆฌ์ผ์ ๋ํ ์์ธํ ์ฝ๋๋ ๋ค์ [Git ๋ ํฌ์งํ ๋ฆฌ](https://github.com/huggingface/transfer-learning-conv-ai)์์ ํ์ธ ํ  ์ ์์ต๋๋ค.



# Personality๋ฅผ ์ง๋ AI ๐ค 

๋ณธ ํ๋ก์ ํธ์ ๋ชฉ์ ์ **Persona** ๋ฅผ ์ง๋ **Conversaional AI** ๋ฅผ ์ ์ํ๋ ๊ฒ์ ๋ชฉํ๋ก ํฉ๋๋ค.

๋ณธ ํ๋ก์ ํธ์ ๋ํ ์์ด์ ํธ(Dialog Agent)๋ ์ด๋ค persona๊ฐ ์ด๋ค ๋ํ ๊ธฐ๋ก(History)์ ์ค๋ชํ๋ ์ง์ ๋ํ Knowledge Base๋ฅผ ๊ฐ๊ณ  ์์ต๋๋ค. ์ฌ์ฉ์๋ก๋ถํฐ ์๋ก์ด ๋ํ๊ฐ ์๋ ฅ๋๋ฉด ๋ํ ์์ด์ ํธ๋ ๋ํ๋ฅผ ๋ถ์ํ์ฌ Persona๋ฅผ ์ง๋ ๋ํ๋ฅผ ์ถ๋ ฅํฉ๋๋ค.

ํ๋ก์ ํธ์ ๊ณํ์ ๋ค์๊ณผ ๊ฐ์ต๋๋ค.

![img](https://miro.medium.com/max/1035/1*sIRX4M3--Qrvo-RLqH-7GQ.png)



### ๋ฅ ๋ฌ๋์ผ๋ก ๋ํ ์์ด์ ํธ๋ฅผ ํ์ต ์ํฌ๋์ ๋ฌธ์ ์ 

- ๋ํ ๋ฐ์ดํฐ ์์ด ์๊ฑฐ๋, ์ ์ฐฝํ๊ฑฐ๋ ๊ด๋ จ์๋ ๋ต๋ณ์ ํ๊ธฐ์ ํ์ต๋์ด ๋ถ์กฑํจ

  โ ์ด์ ๋ํ ํด๊ฒฐ๋ฐฉ์์ผ๋ก 'Smart Beam Search' ๋ฐฉ์์ด ๊ณ ๋ ค ๋์ง๋ง ํด๋น ๊ธ์์๋ **'์ ํ ํ์ต'**(transfer-learning) ์ ํ์ฉํ๋ค.

### ํด๊ฒฐ์ฑ

- ๊ธด ์ฐ์์ ์ผ๋ก ๊ด๋ จ ์๋ ํ์คํธ๋ฅผ ์์ฑํ  ์ ์๋๋ก, ๋งค์ฐ ํฐ ํ์คํธ ๋ง๋ญ์น(very large corpus of text)์ ์ธ์ด ๋ชจ๋ธ์ ๋ฏธ๋ฆฌ ๊ต์กํ๋ ๊ฒ๋ถํฐ ์์ํ๋ค.
- **ํ์ธํ๋**(Fine-Tune)์ ํตํ์ฌ ๋ํ์ ๋ง๊ฒ ๋ฏธ์ธ ์กฐ์ ํ๋ค.

์ธ์ด ๋ชจ๋ธ์ ์ฌ์  ํ์ต(Pretraining)์ผ๋ก ๊ตฌ์ถ ํ๋ ๋ฐฉ๋ฒ์ ์๊ฐ ๋น์ฉ์ด ๋ง์ด ์์๋๊ธฐ์, ์คํ์์ค๋ฅผ ํ์ฉํ๋ค. ๋ํ ๋ฐ์ดํฐ ์์ด ํฌ๊ณ , ์ข์ ๋ฐ์ดํฐ(*The bigger the better*)๋ฅผ ์ฌ์ฉํ๋ฉด ์ข์ง๋ง ๋ฌธ์ฅ ์ฆ, ํ์คํธ๋ฅผ ์์ฑํ๋ ๊ฒ ๋ชฉ์ ์ด๋ค. ๊ทธ๋ฌ๋ฏ๋ก, ์ฌ์ ํ์ต(pretraining)๋ NPL ๋ชจ๋ธ๋ก BERT๋ฅผ ๋ง์ด ์ฌ์ฉํ์ง๋ง, ์๋ฒฝ ๋ฌธ์ฅ(Masking์ด ์๋ ๋ฌธ์ฅ)์์๋ ํ์ต์ด ๋์์ง๋ง, unfinished sentences(Masking์ด ์๋ ๋ฌธ์ฅ)์์๋ ํ์ต์ด ๋์ง ์์๊ธฐ ๋๋ฌธ์ **GPT** & **GPT-2**๋ฅผ ์ฌ์ฉํ๋ค.



## ๐ฆ OpenAI GPT and GPT-2 models

2018๋๊ณผ 2019๋, ์๋  ๋๋ํฌ๋, ์ ํ๋ฆฌ ์ฐ, ์คํ์ ๋๋ฃ๋คAI๋ ๋งค์ฐ ๋ง์ ์์ ๋ฐ์ดํฐ์ ๋ํด ํ๋ จ๋ ๋ ๊ฐ์ง ์ธ์ด ๋ชจ๋ธ์ธ GPT์ GPT-2(Generative Pre-trained Transformer) ์์ฑํ๋ค.

GPT์ GPT-2๋ ๋ค์๊ณผ ๊ฐ์ ๋ฐฉ์์ผ๋ก ์๋ํ๋ค. ***decoder*** ํน์ ***causal*** ์ด๋ผ๊ณ  ๋ถ๋ฆฌ๋ ๋ชจ๋ธ์ด ๋ค์ ๋จ์ด๋ฅผ ์์ธกํ๊ธฐ ์ํด ์ผ์ชฝ ๋ฌธ๋งฅ์ ์ฌ์ฉํ๋ค.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq5a4zj7k2j30w20lwjs5.jpg" alt="https://miro.medium.com/max/2077/1*YmND0Qj8O6b35J1yU_CPKQ.png" style="zoom: 50%;" />

Decoder/Casual ํธ๋์คํฌ๋จธ๋ ์ผ์ชฝ์์ ๋ถํฐ ๋ค์ ๋จ์ด๋ฅผ ์์ฑ์ ์์ธกํ๋ค. [์ดํ์ ๋ฉ์ปค๋์ฆ (Attention Mechanism)](https://www.notion.so/Attention-Mechanism-d6ede65d09ee4be9b20fc4f19e074d13) ์ ๊ธฐ๋ฐ์ผ๋ก ํ๋ค. ์ดํ์ ๋งค์ปค๋์ฆ์ ๋ค์ ์ฌ์ดํธ์์ ์์ธํ ๋ด์ฉ์ ์ฐพ์ ๋ณผ ์ ์๋ค .[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

๋ณธ ํ๋ก์ ํธ๋ฅผ ์ํํ๊ธฐ ์ํด์, ์ธ์ด ๋ชจ๋ธ(Language Model)์ ๋จ์ํ ์๋ ฅ ์ํ์ค(Input Sequence)๋ฅผ ์๋ ฅ์ผ๋ก ๋ฐ์ ๋ฟ๋ง ์๋๋ผ, ์๋ ฅ ์ํ์ค ๋ค์์ ์ด์ด์ง๋ ํ ํฐ์ ๋ํ ์ดํ์ ๋ํ ํ๋ฅ  ๋ถํฌ๋ฅผ ์์ฑ ํด์ผํ๋ค. ์ธ์ด ๋ชจ๋ธ์ ๋๊ฐ ์์ ๊ทธ๋ฆผ์ ํ์๋ ๊ฒ์ฒ๋ผ ๊ธด ์๋ ฅ ์ํ์ค์์ ๊ฐ ํ ํฐ์ ๋ฐ๋ฅด๋ ํ ํฐ์ ์์ธกํ์ฌ ๋ณ๋ ฌ ๋ฐฉ์์ผ๋ก ํ๋ จ๋๋ค.

๋๊ท๋ชจ ๋ง๋ญ์น์์ ์ด๋ฌํ ๋ชจ๋ธ์ ์ฌ์  ๊ต์กํ๋ ๊ฒ์ ๋น์ฉ์ด ๋ง์ด ๋๋ ์์์ด๊ธฐ ๋๋ฌธ์ Open AI์์ Pre-trained๋  ๋ชจ๋ธ๊ณผ ํ ํฐ๋ผ์ด์ ๋ฅผ ์ฌ์ฉํ๋ค. Tonkenizer๋ ์๋ ฅ ๋ฌธ์์ด์ ํ ํฐ(๋จ์ด/ํ์ ๋จ์ด)์ผ๋ก ๋ถํ ํ๊ณ , ์ด๋ฌํ ํ ํฐ์ ๋ชจ๋ธ ์ดํ๋ฅผ ๋ณํํ๋ค.





## ํ๋ก์ ํธ ๋ชฉ์ 

- ํ ํฐ ์ํ์ค(a sequence of tokens)๋ฅผ ์ธํ์ผ๋ก ๋ฐ๊ณ ,
- **์๋ ฅ ์์์ ๋ฐ๋ผ ๋ค์ ํ ํฐ์ ์ดํ์ ๋ํ ํ๋ฅ  ๋ถํฌ๋ก ์์ฑ**
- ์ธ์ด ๋ชจ๋ธ์ ์ผ๋ฐ์ ์ผ๋ก ์์ ๊ทธ๋ฆผ์์ ์ค๋ชํ ๊ฒ์ฒ๋ผ ๊ธด ์๋ ฅ ์ํ์ค์์ ๊ฐ ํ ํฐ์ ๋ฐ๋ฅด๋ ํ ํฐ์ ์์ธกํ์ฌ **๋ณ๋ ฌ ๋ฐฉ์**์ผ๋ก ํ๋ จ๋ฉ๋๋ค.

๋ง๋ญ์น๊ฐ ํฐ ๊ฒฝ์ฐ ํ ํฐํ์ ๋ง์ ๋น์ฉ์ด ๋ฐ์ํ๋ค. ๊ทธ๋์ OpenAI์์ ์ฌ์ ํ์ต๋ tokenizer๋ก ์ฐ๋ฆฌ ๋ชจ๋ธ์ ์ ์ฉํด ๋ณด์๋ค.

[pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) OpenAI GPT ๋ชจ๋ธ์ ์ฌ์ฉํ์ฌ ๋ชจ๋ธ๊ณผ ํ ํฌ๋์ด์ ๋ฅผ ๋ถ๋ฌ์จ๋ค.

```python
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
```

*OpenAI GPT Double Heads Model* ๋ผ๋ ๋ชจ๋ธ๋ฅผ ๋ถ๋ฌ์ด



## ๐ป ๋ํ์์ญ์ ์ธ์ด ๋ชจ๋ธ ์ ์ฉ

---

์ฐ๋ฆฌ ๋ชจ๋ธ์ **๋จ์ผ์๋ ฅ(Single Input)** ์ ํ๋ จ๋์๋ค: ์ผ๋ จ์ ๋จ์ด๋ค(a sequence of words)

์ถ๋ ฅ ์ํ์ค๋ฅผ ์์ฑํ๋ ์ฌ๋ฌ ์ ํ์ **์ปจํ์คํธ**(***Context***)๋ฅผ ์ฌ์ฉํ๋ค:

- ํ ๊ฐ ๋๋ ์ฌ๋ฌ ๊ฐ์ ๊ฐ์ธ ์ค์  ๋ฌธ์ฅ,
- ์ฌ์ฉ์๋ก๋ถํฐ ์ต์ํ ๋ง์ง๋ง ๋ง์ ํ ๋ํ ๊ธฐ๋ก,
- ์ถ๋ ฅ ์ํ์ค ๋จ์ด๋ฅผ ๋จ์ด๋ณ๋ก(word by word) ์์ฑํ ์ดํ ์ด๋ฏธ ์์ฑ๋ ์ถ๋ ฅ ์ํ์ค์ ํ ํฐ(?)

> ์ด๋ป๊ฒ ์์ ๊ฐ์ ๋ค์ํ ๋งฅ๋ฝ(***Context***)์ ๊ณ ๋ คํ ์๋ ฅ์ ๋ง๋ค ์ ์์๊น?

๋ฌธ๋งฅ(Context) ์ธ๊ทธ๋จผํธ(๋ถ๋ถ)๋ฅผ ํ๋์ ์ํ์ค๋ก **์ฐ๊ฒฐ(Concatenate)**์์ผ, ๊ทธ ๋๋ต์ ๋ง์ง๋ง์ ๋๋ ๊ฒ์ด๋ค. ๊ทธ๋ฐ ๋ค์, ๋ค์ ์ํ์ค๋ฅผ ๊ณ์ ์์ฑํ์ฌ ํ ํฐ์ผ๋ก ์๋ต ํ ํฐ ์์ฑํ  ์ ์๋ค.

![https://miro.medium.com/max/4711/1*RWEUB0ViLTdMjIQd61_WIg.png](https://tva1.sinaimg.cn/large/008i3skNgy1gq5afy5h9aj320p0owwgz.jpg)

(ํ๋์): ๋ณํฉ๋ ํ๋ฅด์๋, (ํํฌ), (๋น์): ์ด์  ๋ํ ํ์คํ ๋ฆฌ

### ํ์ง๋ง, ์์ ๊ฐ์ ๋ฐฉ์์๋ ๋ ๊ฐ์ง ๋ฌธ์ ์ ์ด ์กด์ฌ:

- "*ํธ๋์คํฌ๋จธ๋ ์ ๊ทธ๋ฆผ๊ณผ ๊ฐ์ด ์๊น๋ก ๊ตฌ๋ถ ํ์ง ๋ชปํ๋ค."* ๊ตฌ๋ถ ํ ํฐ(The delimiter tokens)์ ๊ฐ ๋จ์ด๊ฐ ์ด๋ค ๋ถ๋ถ์ ์ํ๋์ง ๊ฐ๋จํ ์๋ ค์ค๋ค. ์๋ฅผ ๋ค์ด, "NYC"๋ผ๋ ๋จ์ด๋ ์ฐ๋ฆฌ์ ๊ทธ๋ฆผ์์ ํ๋์์ผ๋ก ํ์๋์ง๋ง, ์ฐ๋ฆฌ์ ๋ชจ๋ธ์ ๊ตฌ๋ถ์(delimiter)๋ก๋ง Persona์ธ์ง, History์ธ์ง ์ฌ๋ถ๋ฅผ ์ถ์ถํ๋๋ฐ ์ด๋ ค์์ด ์๋ค: ๊ทธ๋ฌ๋ฏ๋ก **์ธ๊ทธ๋จผํธ์ ๋ํ ๋ ๋ง์ ์ ๋ณด๋ฅผ ์ถ๊ฐํด์ผ ํ๋ค.**
- *"ํธ๋์คํฌ๋จธ๋ ๋จ์ด์ ์์น๋ฅผ ์ ์ ์๋ค"* ์ฌ๊ธฐ์ ์ดํ ์(Attention)์ Dot-Product Attention์ ์ฌ์ฉํ๋ค. ๊ทธ๋์ **์ฐ๋ฆฌ๋ ํ ํฐ ๋ง๋ค ์์น ์ ๋ณด๋ฅผ ์ถ๊ฐ ํด์ผํ๋ค.**
- Dot-Product Attention์ ๋ํ ์ ๋ณด

    [์ดํ์ ๋ฉ์ปค๋์ฆ๊ณผ transfomer(self-attention)](https://medium.com/platfarm/%EC%96%B4%ED%85%90%EC%85%98-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98%EA%B3%BC-transfomer-self-attention-842498fd3225)

    [์ํค๋์ค](https://wikidocs.net/22893)

์ด๋ฅผ ํด๊ฒฐ ํ๊ธฐ ์ํด ๋จ์ด, ์์น ๋ฐ ์ธ๊ทธ๋จผํธ(***word, position and segments***)์ ๋ํด ์๋ ฅ์ ๋ฐ๋ 3๊ฐ์ ๋ณ๋ ฌ ์๋ ฅ ์ํ์ค๋ฅผ ๋ง๋ค๊ณ ,  ์ด๋ค์ ํ๋์ ๋จ์ผ ์ํ์ค(Single Sequence)๋ก ๊ฒฐํฉํ๋ค. ์ฆ, ๋จ์ด ์์น ๋ฐ ์ธ๊ทธ๋จผํธ ์๋ฒ ๋ฉ์ ์ธ ๊ฐ์ง ์ ํ์ ํฉ์ฐํ๋ ๊ฒ์๋๋ค.

![https://miro.medium.com/max/4711/1*r7vi6tho6sfpVx-ZQLPDUA.png](https://tva1.sinaimg.cn/large/008i3skNgy1gq5ag3gx7ij320p0py77x.jpg)

## ์คํ

---

์ฒซ๋ฒ์งธ๋ก, ๊ตฌ๋ถ ๊ธฐํธ ๋ฐ ์ธ๊ทธ๋จผํธ ํ์๊ธฐ์ *ํน์ ํ ํฐ(special Token)*์ ์ถ๊ฐํ๋ค. ์ด ํน์ ํ ํฐ์ ์ฌ์  ํ์ต ๋ชจ๋ธ์ ํฌํจ ๋์ง ์์์, ์๋ก ํ๋ จํ๊ณ , ์๋ฒ ๋ฉํ ํ์ฌ ์์ฑํ๋ค. (**create** and **train new embeddings)-**  [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) ์ฌ๊ธฐ์์ ์ฝ๊ฒ ๊ฐ๋ฅํจ.

```python
# ๋ค์๊ณผ ๊ฐ์ด 5 ๊ฐ์ง special tokens์ ์ฌ์ฉํจ:
# - <bos> the sequence์ ์ฒ์์ ๊ฐ๋ฅดํด
# - <eos> the sequence์ ๋์ ๊ฐ๋ฅดํด
# - <speaker1> ์ ์ ์ ๋ฐํ(utterance) ์ฒซ๋ถ๋ถ์ ๊ฐ๋ฅดํด
# - <speaker2> ์ฑ๋ด์ ๋ฐํ(utterance) ์ฒซ๋ถ๋ถ์ ๊ฐ๋ฅดํด
# - <pad> as a padding token to build batches of sequences
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]

# We can add these special tokens to the vocabulary and the embeddings of the model:
tokenizer.set_special_tokens(SPECIAL_TOKENS)
model.set_num_special_tokens(len(SPECIAL_TOKENS))
```

```python
from itertools import chain

# Let's define our contexts and special tokens
persona = [["i", "like", "playing", "football", "."],
           ["i", "am", "from", "NYC", "."]]
history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]
reply = ["great", "to", "hear"]
bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"

def build_inputs(persona, history, reply):

words, segments, position, sequence = build_inputs(persona, history, reply)

# >>> print(sequence)  # Our inputs looks like this:
# [['<bos>', 'i', 'like', 'playing', 'football', '.', 'i', 'am', 'from', 'NYC', '.'],
#  ['<speaker1>', 'hello', 'how', 'are', 'you', '?'],
#  ['<speaker2>', 'i', 'am', 'fine', 'thanks', '.'],
#  ['<speaker1>', 'great', 'to', 'hear', '<eos>']]

# Tokenize words and segments embeddings:
words = tokenizer.convert_tokens_to_ids(words)
segments = tokenizer.convert_tokens_to_ids(segments)
```





## ๐ Multi-tasks losses

---

์ฐ๋ฆฌ๋ ์ด์  ์ฌ์  ํ๋ จ๋ ๋ชจ๋ธ์ ์ด๊ธฐํํ๊ณ  ํ๋ จ ์๋ ฅ์ ๊ตฌ์ถํ์ผ๋ฉฐ, ๋จ์ ๊ฒ์ ํ์ธ ํ๋ ์ค์ ์ต์ ํํ  ์์ค์ ์ ํํ๋ ๊ฒ๋ฟ์ด๋ค

> ๋ฌธ์ฅ ์์ธก(Next-sentence prediction)์ ์ํด ์ธ์ด์ ๊ฒฐํฉ๋ multi-task loss์ ์ฌ์ฉํ๋ค.

๋ฌธ์ฅ ์์ธก ๋ชฉํ๋ BERT ์ฌ์ ํ์ต ๋ถ๋ถ์ด๋ค. ๊ทธ๊ฒ์ ๋ฐ์ดํฐ ์ธํธ์์ ๋ฌด์์๋ก *distactor*(์ ๋ต ์ด์ธ์ ์ ํ์ง)๋ฅผ ์ถ์ถํ๊ณ , ์๋ ฅ ์ํ์ค๊ฐ gold reply(?)๋ก ๋๋๋์ง ์๋๋ฉด distactor๋ก ๋๋๋์ง ๊ตฌ๋ณํ๊ธฐ ์ํ ๋ชจ๋ธ์ ํ๋ จํ๋ ๊ฒ์ผ๋ก ๊ตฌ์ฑ๋๋ค. Local ๋ฌธ๋งฅ(Context) ์ด์ธ์, Global ์ธ๊ทธ๋จผํธ ์๋ฏธ๋ฅผ ์ดํด๋ณด๋ ๋ชจ๋ธ์ ๊ต์กํ๋ค.

![https://miro.medium.com/max/5326/1*945IpgUS9MGLB6gchoQXlw.png](https://miro.medium.com/max/5326/1*945IpgUS9MGLB6gchoQXlw.png)

Multi-Task Training ๋ชฉ์  - ๋ชจ๋ธ์ ์ธ์ด ๋ชจ๋ธ๋ง ์์ธก์ ์ํด ๋ ๊ฐ์ ํค๋๋ฅผ ์ ๊ณต(์ค๋ ์ง), ์์ธก ๋ฌธ์ฅ ๋ถ๋ฅ๊ธฐ(ํ๋์)

### Total Loss๋ ๋ค์๊ณผ ๊ฐ์ด ๊ณ์ฐ๋๋ค. **language modeling loss** ๊ณผ **next-sentence prediction loss๋ฅผ ํตํด(์ดํด ์๋จ)**

- **Language modeling:** we project the hidden-state on the word embedding matrix to get logits and apply a cross-entropy loss on the portion of the target corresponding to the gold reply (green labels on the above figure).
- **Next-sentence prediction:** we pass the hidden-state of the last token (the end-of-sequence token) through a linear layer to get a score and apply a cross-entropy loss to classify correctly a gold answer among distractors.



## ๐ป Decoder ์ธํ

์ธ์ด ์์ฑ์ ์ํด์๋ **greedy-decoding** ๊ณผ **beam-search** ๋ฐฉ๋ฒ์ ์ฃผ๋ก ์ฌ์ฉํ๋ค. 

**Greedy-decoding**์ ๋ฌธ์ฅ์ ๋ง๋๋ ๊ฐ์ฅ ๊ฐ๋จํ ๋ฐฉ๋ฒ์ด๋ค. ๋งค ๋จ๊ณ๋ง๋ค ์์์ ๋ ํ ํฐ์ ๋๋ฌํ  ๋๊น์ง ๋ชจ๋ธ์ ๋ฐ๋ผ ๊ฐ์ฅ ๊ฐ๋ฅ์ฑ์ด ๋์ ๋ค์ ํ ํฐ์ ์ ํํ๋ค. **Greedy-decoding**์ ํ ๊ฐ์ง ์ํ์ ๊ฐ๋ฅ์ฑ์ด ๋งค์ฐ ๋์ ํ ํฐ์ด ๋ฎ์ ํ ํฐ ๋ค์ ์จ์ด ์๋ค๊ฐ ๋์น  ์ ์๋ค๋ ๊ฒ์ด๋ค.

**Beam-search**๋ ๋จ์ด๋ณ๋ก ๊ตฌ์ฑํ๋ ๋ช ๊ฐ์ง ๊ฐ๋ฅํ ์์์ Beam(ํ๋ผ๋ฏธํฐ: ๋๋น์ ์)๋ฅผ ์ ์งํจ์ผ๋ก์จ ์ด ๋ฌธ์ ๋ฅผ ์ํํ๋ ค๊ณ  ๋ธ๋ ฅํ๋ค. ๊ทธ ๊ณผ์ ์ด ๋๋๋ฉด Beam ์ค์์ ๊ฐ์ฅ ์ข์ ๋ฌธ์ฅ์ ๊ณ ๋ฅธ๋ค. ์ง๋ ๋ช ๋ ๋์ ๋น ๊ฒ์์ ๋ํ ์์๋ฅผ ํฌํจํ ๊ฑฐ์ ๋ชจ๋  ์ธ์ด ์์ฑ ์์์์ ํ์ค ๋์ฝ๋ฉ ์๊ณ ๋ฆฌ์ฆ์ด์๋ค.

### ๊ธฐ์กด ๋ฌธ์ ์ 

์ต๊ทผ ๋ผ๋ฌธ(Ari Holtzman et al.)์ ๋ฐ๋ฅด๋ฉด "๋น ๊ฒ์๊ณผ ํ์์ค๋ฌ์ด ํด๋๋ฒ์ ์ฌ์ฉํ์ฌ ์์ฑ๋ ํ์คํธ์ ๋จ์ด ๋ถํฌ๋ ์ธ๊ฐ์ด ๋ง๋  ํ์คํธ์ ๋จ์ด ๋ถํฌ์ ๋งค์ฐ ๋ค๋ฅด๋ค." ๋ผ๊ณ  ํ๋ค. ๋ถ๋ชํ ๋น ๊ฒ์๊ณผ ํ์์ค๋ฌ์ด ํด๋์ ๋ํ ์์คํ์ ๋งฅ๋ฝ์์ [7, 8]์์๋ ์ธ๊ธ๋์๋ฏ์ด ์ธ๊ฐ ๋ฐํ์ ๋ฌธ์ ๋ถํฌ ์ธก๋ฉด์ ์ฌํํ์ง ๋ชปํ๋ค.

![https://miro.medium.com/max/2830/1*yEX1poMDsiEBisrJcdpifA.png](https://miro.medium.com/max/2830/1*yEX1poMDsiEBisrJcdpifA.png)

*์ผ์ชฝ: ์ฌ๋์ด ์์ฑํ ํ ํฐ๊ณผ GPT-2๋ฅผ ์ฌ์ฉํ ๋น ๊ฒ์์ ํ ๋น๋ ํ๋ฅ (๋น ๊ฒ์์ผ๋ก ์ฌํ๋์ง ์์ ์ธ๊ฐ ํ์คํธ์ ๊ฐํ ์ฐจ์ด๋ฅผ ์ฐธ๊ณ ) ์ค๋ฅธ์ชฝ: ์ธ๊ฐ๊ณผ ๊ธฐ๊ณ๋ก ์์ฑ๋ ํ์คํธ์ N-๊ทธ๋จ ๋ถํฌ(Greedy/Beam Search).*

### ํด๊ฒฐ์ฑ

ํ์ฌ Beam Search/Greedy ๋์ฝ๋ฉ์ ์ฌ์ฉํ  ์ ์๋ ๋ฐฉ๋ฒ๋ก ์ **top-k**์ **nuclear**(๋๋ **top-p**) ์ํ๋ง์ด๋ค. ์ด ๋ ๋ฐฉ๋ฒ๋ก ์ ๋ถํฌ๋ฅผ ํํฐ๋งํ ํ ๋ค์ ํ ํฐ ๋ถํฌ์์ ํ๋ณธ์ ์ถ์ถํ์ฌ ๋์  ํ๋ฅ ์ด ์๊ณ๊ฐ(nucleus/top-p) ๋ณด๋ค ๋์ ์๋ ์์ ํ ํฐ(top-k) ๋๋ ์์ ํ ํฐ๋ง ์ ์งํ๋ ๊ฒ์ด๋ค.

๊ทธ๋ฌ๋ฏ๋ก, **top-k** ์ **nucleus/top-p** ์ํ๋ง์ ๋์ฝ๋๋ก ์ฌ์ฉํ๊ธฐ๋ก ๊ฒฐ์  ํ์๋ค.



## ๊ฒฐ๋ก 

๋ณธ ํฌ์คํ์์ ์ค๋ช ํ ๊ฒ์ฒ๋ผ, Hugging Face์์๋ **Conversational AI**๋ฅผ ๊ตฌํ ํ๊ธฐ ์ํด ๋์ฉ๋ ์ธ์ด ๋ชจ๋ธ(large-scale language model)์ธ OpneAI์ **GPT-2**๋ฅผ ์ฌ์ฉํ๋ค

๋ณธ ํ๋ก์ ํธ์ ๋ฐ๋ชจ์ ์์ธํ ์ฝ๋๋ ๋ค์ ๋งํฌ์์ ์ฐพ์ ๋ณผ ์ ์๋ค.

- [๋ฐ๋ชจ](http://convai.huggingface.co/)
- [Pre-trained ๋ชจ๋ธ](https://github.com/huggingface/transfer-learning-conv-ai)



## References

- [^](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313#c7f2) *Importance of a Search Strategy in Neural Dialogue Modelling* by Ilya Kulikov, Alexander H. Miller, Kyunghyun Cho, Jason Weston (http://arxiv.org/abs/1811.00907)

- [^](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313#5aac) *Correcting Length Bias in Neural Machine Translation* by Kenton Murray, David Chiang (http://arxiv.org/abs/1808.10006)

- [^](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313#5aac) *Breaking the Beam Search Curse: A Study of (Re-)Scoring Methods and Stopping Criteria for Neural Machine Translation* by Yilin Yang, Liang Huang, Mingbo Ma (https://arxiv.org/abs/1808.09582)

- [^](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313#97f9) *Hierarchical Neural Story Generation* by Angela Fan, Mike Lewis, Yann Dauphin (https://arxiv.org/abs/1805.04833)

- [^](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313#97f9) *Language Models are Unsupervised Multitask Learners* by Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever (https://openai.com/blog/better-language-models/)

- [^](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313#6ea4) *The Curious Case of Neural Text Degeneration* by Ari Holtzman, Jan Buys, Maxwell Forbes, Yejin Choi (https://arxiv.org/abs/1904.09751)

- *Retrieve and Refine: Improved Sequence Generation Models For Dialogue* by Jason Weston, Emily Dinan, Alexander H. Miller (https://arxiv.org/abs/1808.04776)

- [^](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313#6ea4) *The Second Conversational Intelligence Challenge (ConvAI2)* by Emily Dinan et al. (https://arxiv.org/abs/1902.00098)