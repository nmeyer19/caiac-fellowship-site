---
order: 2
label: "Week 2"
title: "ARENA 1.1 (Transformers from Scratch)"
start: 2026-10-04
end: 2026-10-10
arena:
  section: "1.1"
  url: https://learn.arena.education/chapter1_transformer_interp/01_transformers/
ready: true
---

## This week's exercises

| Section | Exercise | Importance | Time |
| --- | --- | --- | --- |
| 1.1 |  | /5 |  min |


## Background and Overview
In 2017, researchers at Google made a breakthrough in machine translation with the (now canonical) paper [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Long story short, they mushed together a bunch of stuff that had been building up over the past couple decades into a new network architecture called the transformer that would prove to change the trajectory of language modeling and machine learning in general. 

We saw some of these things last week with CNNs and residual networks. Today, we build off of these ideas and explore the full structure of the transformer. Notably, the version of the transformer from this paper is a little different from what we'll be building. It's what's called an [encoder-decoder model](https://www.ibm.com/think/topics/encoder-decoder-model), but most of today's LLMs, including GPT-2 (what we're building), are decoder-only models.

<img src="/images/ch1.1_alluneed.jpeg" alt="The original encoder-decoder transformer from Attention Is All You Need" style="width: 24rem">

Some other changes, which we'll get into more below, are that GPT-2-like models don't have what's called cross-attention (because there's no encoder to do cross-attention with), do the layernorm *before* each sublayer, positional embeddings are trained/learned, and use [GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html) instead of [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html) (which, for all intents and purposes, are basically the same thing, GELU is just ~better~).

<img src="/images/ch1.1_transformeroverview.png" alt="Ultra-high-level transformer overview" style="width: 40rem">

Here's a high-level overview of what's going on. We start with an input sequence of text. This text first gets turned into tokens which are fed into our transformer, where these tokens get manipulated this way and that, over and over and all in parallel until eventually some stuff comes out the other end: probability distributions over our model's vocabulary representing what it thinks the next token should be given the tokens that have come through up to this point.

Going a step deeper, here is the rough path by which a (single) sequence of tokens flows through a transformer:
- Each token (initially represented by an integer token ID corresponding to a token in our vocabulary) is first embedded (i.e. linearly transformed) via our embedding matrix ($W_E \in \mathbb{R}^{v \times d}$) and positional embedding matrix ($W_P \in \mathbb{R}^{n_{ctx} \times d}$), and the two results are added together to give a $d$-dimensional vector. Across the sequence, we now have an $[s \times d]$ tensor floating down the network's *residual stream* (which, in this example, I like to think about as $s$ many $d$-dimensional vectors, i.e. one vector for each token in our sequence).
- Then we arrive at the first **transformer block**. Here, a copy of our tensor keeps floating down the residual stream, while a separate copy goes into the block.
- Inside the block, 
  - Each token's vector is first layer-normalized. 
  - Then, the sequence tensor is fed into each attention head (in parallel). 
  - In each head,
    - Each token vector is transformed into a key, query, and value vector.
    - Attention scores are calculated over the keys and queries. 
    - A "causal mask" is then applied to ensure we can't cheat by looking ahead to future tokens in the sequence. 
    - Masked attention scores are turned into probabilities and multiplied with the value vectors. 
    - The result is projected back up to `d_model` with an output matrix ($W_O$).
  - The outputs of each head are summed together and added back into the residual stream.
  - One copy of the vectors keeps floating down the stream while another comes back into the block and is layer-normalized *again* and fed into a simple feed-forward network (MLP), where
    - Each vector is projected up to a higher dimension (`d_mlp`) and an activation function is applied (`GELU`).
    - We then project each vector back down to `d_model` and add this back into the residual stream.
- We repeat this for each transformer block.
- We do one final LayerNorm, multiply our vectors by an unembedding matrix ($W_U \in \mathbb{R}^{d \times v}$) and get our `logits` (a score for each token in our vocab).
- We SoftMax our logits to normalize and turn these scores into a probability distribution. We sample from the distribution at the last position to get our next token!


## Tokenization
<img src="/images/ch1.1_tokenization.png" alt="BPE tokenization example" style="width: 36rem">

This pretty much sums it up for our purposes. A tokenizer breaks up text into tokens, each represented by a unique integer ID. These token IDs are what's then actually fed into our transformer. 

GPT-2 also uses BPE, and its nuances aren't super important here. If you want to learn more, you can check out [this](https://huggingface.co/learn/llm-course/en/chapter6/5) explainer from HuggingFace. 

[Here's](https://tiktokenizer.vercel.app/?model=gpt2) a fun tokenizer interface to play with and build some intuitions around. This demo uses the GPT-2 tokenizer. You can also check out [Karpathy's video](https://www.youtube.com/watch?v=zduSFxRajkE) on building a tokenizer from scratch, if you're so inclined. 


## Embeddings
So, we now have a list of token IDs, which are just integers—a list with length $n=d\_sequence$. We can then think of this list as a vector, which we'll call our input vector,

$$t = (t_1, ..., t_n) \in \{1, ..., d\_vocab\}^{d\_sequence}$$

which is fed into our transformer. For now, we'll just look at a single token, $t_i$. The first thing it interacts with is the embedding matrix, 

$$W_E \in \mathbb{R}^{d\_vocab \times d\_model}$$

Which we'll want to **index into**. In code, we index directly (`W_E[tokens]`). But to write this as math, we convert the integer $t_i$ into a one-hot representation $e_t \in \mathbb{R}^{d\_vocab}$. We note that $t_i$ is the token ID $t \in \{1, ..., d\_vocab\}$ corresponding to a word in our vocabulary. $e_t$ is then the vector with a $1$ at position $t$ and $0$ everywhere else.

$$x_E = e^\top_{t}W_E$$

Through this, we're basically performing a row lookup into our embedding matrix using the one-hot representation. Each row in our embedding matrix is something like a rich multidimensional representation of the token carrying important semantic information about that word.

So in sum:

One hot: $t_i \in \{1, ..., d\_vocab\} \to e_t \in \mathbb{R}^{d\_vocab}$

Embedding: $x_E = e^\top_{t}W_E \in \mathbb{R}^{d\_model}$


## Positional Embeddings
Our positional embedding follows much of the same process, but we index by each token's **position** in the sequence, not by its token ID. So every token at position 3 gets the same positional embedding, whatever the token is. $W_P \in \mathbb{R}^{n\_ctx \times d\_model}$ has one row for each position up to the maximum context length $n\_ctx$ (1024 for GPT-2), so the model can't process sequences longer than that.

So for the token at position $i$, we have the positional one-hot encoding $e_i \in \mathbb{R}^{n\_ctx}$, 

which we then use to get the positional embedding

$$x_P = e^\top_{i}W_P \in \mathbb{R}^{d\_model}$$


## The Residual Stream 
Putting these together, we get the start of our residual stream for token $t_i$:

$$x = x_E + x_P = e^\top_{t}W_E + e^\top_{i}W_P \in \mathbb{R}^{d\_model}$$

We can think of each token in our sequence as having its "own" residual stream (which will come in handy later when we talk about attention). But, if we want to represent these all together, as we'll often have to in our code, we basically just have $d\_sequence$ of these $d\_model$ vectors stacked together:

$$X = X_E + X_P = TW_E + PW_P \in \mathbb{R}^{d\_sequence \times d\_model}$$

where $T \in \mathbb{R}^{d\_sequence \times d\_vocab}$ stacks the token one-hots $e_t$, and $P \in \mathbb{R}^{d\_sequence \times n\_ctx}$ stacks the position one-hots $e_i$.

Each layer from here will read a copy from the stream(s) and write its output back.


## LayerNorm
<img src="/images/ch1.1_ln.png" alt="LayerNorm" style="width: 40rem">

GPT-2 (and most transformers today) do pre-layer normalization, rather than post- like the original Attention Is All You Need paper did. The reasoning behind this is basically that empirically it's been shown to enable stabler training and creates a "cleaner" residual stream, allowing us to treat it as this linear and additive communication channel between layers, in which it's theoretically easier to trace and attribute particular contributions to particular places. 

The main alternative to layer normalization is batch normalization. As the names suggest, the key difference is the axis over which we're normalizing: LayerNorm normalizes each token's vector across its `d_model` features, while BatchNorm normalizes each feature across the batch. The nuances here aren't that important, and the moral of the story is that empirically LayerNorm just works better, and that's why we use it. If I were to mention one thing, it's that LayerNorm keeps positions independent, and BatchNorm can leak information between token positions in a way that might interfere with our causal mask and the process of autoregressive sampling. 

Back to the equation: for a particular token's residual stream vector, we're subtracting the mean across its `d_model` features, dividing by the standard deviation (technically the square root of the variance plus a tiny epsilon term to ensure we never divide by zero and break our model). We then apply a learned scale parameter and add a bias term. 

As with many other things, PyTorch has a handy `torch.nn.LayerNorm` module for us (though in the exercises you'll implement it yourself).

## Attention
As we're all aware, attention is the crux of the transformer architecture—both in terms of what it's able to accomplish and in terms of difficulty in understanding...

### Single-Head Attention
We'll start by thinking through what attention looks like in a single attention head. First, let's recall the shape of what we're processing. We'll focus on a single token in a sequence, and then after first building up our intuitions with this, scale this up to how things are actually processed (and how you'll be writing the code). 

<img src="/images/ch1.1_qkvcartoon.png" alt="QKV Cartoon" style="width: 44rem">

We have our (normalized) residual stream at a given layer $l$ coming into our attention head:

$$X_l \in \mathbb{R}^{d\_sequence \times d\_model}$$

Which, again, we can think of as `d_sequence` vectors $x_t \in \mathbb{R}^{d\_model}$ stacked together.

In a given attention head, we have three weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{d\_model \times d\_head}$ and their corresponding bias vectors $b_Q, b_K, b_V \in \mathbb{R}^{d\_head}$, which we use to derive each token's query, key, and value vectors:

$$q_t = x_tW_Q + b_Q \in \mathbb{R}^{d\_head}$$
$$k_t = x_tW_K + b_K \in \mathbb{R}^{d\_head}$$
$$v_t = x_tW_V + b_V \in \mathbb{R}^{d\_head}$$

Using the queries and keys, we then calculate attention scores for every query, key combo:

$$score_{i,j} = q_ik_j^\top \times \frac{1}{\sqrt{d\_head}}$$

For some $i,j \in \{1, ..., d\_sequence\}$. We then do this $\forall \space i,j$, to get a matrix of attention scores 

$$S = \frac{QK^\top}{\sqrt{d\_head}} \in \mathbb{R}^{d\_sequence \times d\_sequence}$$

where $Q, K, V \in \mathbb{R}^{d\_sequence \times d\_head}$ are the $q_t$, $k_t$, $v_t$ vectors stacked together.

We can think of each attention score representing the compatibility between a given query and key, i.e. how well token j's key matches token i's query (its "question"). We divide by the square root of the head's dimensionality just to keep the scores in a range that doesn't saturate the softmax that follows. 

But notice, there's an issue with this: when $j > i$, a token's query is being answered by a token that comes after it. That's a big no-no! If we let this happen, then the model would just learn something super trivial like "ask the token in the next position what it is and then just put all of the probability mass onto that token in the vocab for our prediction," get perfect accuracy on its training data, and be utterly useless.

To fix this, we add a **causal mask** to our score matrix, where we set the score to $-\infty$ for every (query, key) pair where the key has a higher index than the query. After the softmax, these pairs get a weight of exactly $0$.

<img src="/images/ch1.1_causalmask.png" alt="Causal Mask" style="width: 48rem">

As shown in the image, we then softmax over the key dimension, where each row becomes a probability vector over which we basically get weights for how much a given token should "attend to" the other tokens that have come before it. This pattern of weights is known as the "attention pattern." So, we now have a lower-triangular matrix 

$$A = softmax(mask(S)) \in \mathbb{R}^{d\_sequence \times d\_sequence}$$

We then multiply our attention pattern by our value vectors, which we recall are `d_sequence` stacked `d_head` vectors. This gives $Z = AV \in \mathbb{R}^{d\_sequence \times d\_head}$: each token's output is a weighted sum of the value vectors of the tokens it attends to. We then project back out using $W_O \in \mathbb{R}^{d\_head \times d\_model}$ to the residual stream's dimensionality:

$$attention\_out = ZW_O + b_O = AVW_O + b_O \in \mathbb{R}^{d\_sequence \times d\_model}$$

(With more than one head, $b_O$ is added only once, after the heads' outputs are summed.)

<img src="/images/ch1.1_sha.png" alt="Single-head attention: resid_pre to keys, queries and values, attention scores, causal mask, softmax, z, and W_O to attn_out, with parameter shapes" style="width: 34rem">

#### QK and OV Circuits

### Multi-Head Attention
<img src="/images/ch1.1_mha.png" alt="Multi-head attention: per-head keys, queries and values, attention scores and probabilities, z, per-head results summed over the head dimension plus b_O to give attn_out, with parameter shapes" style="width: 30rem">


## Multi-Layer Perceptron (MLP)
<img src="/images/ch1.1._mlpcartoon.png" alt="MLP cartoon: each token does its own internal processing, with no communication" style="width: 18rem">

<img src="/images/ch1.1_mlp.png" alt="MLP: linear layer d_model to d_mlp, GELU, linear layer d_mlp to d_model, shown inside a transformer block" style="width: 40rem">


## Output


## Training Objective


## Scaling Up


## Further Resources
1. [ARENA Lecture on Transformers](https://youtu.be/11Z50mi8dSg?si=8fFq3aWCHDmXU6Ik)
1. [What is a Transformer?](https://www.youtube.com/watch?v=bOYE6E8JrtU&t=1s)
1. [Implementing GPT-2 From Scratch](https://youtu.be/dsjUDacBw8o?si=5W9cqvOKznmguPBX)
1. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
1. [A Walkthrough of A Mathematical Framework for Transformer Circuits](https://youtu.be/KV5gbOmHbjU?si=eK80cWcC-jgt2r0O)
1. [Language Models are Unsupervised Multitask Learners (GPT2 Paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
1. [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)