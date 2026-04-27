---

## Bag of Words (BoW) Model

- The Bag of Words model is a simple and widely used method for representing text data in natural language processing (NLP).
- In BoW, a text (such as a sentence or document) is represented as a multiset (bag) of its words, disregarding grammar and word order but keeping multiplicity.
- Each unique word in the corpus becomes a feature, and the value is typically the count (or frequency) of that word in the document.

### Drawbacks of Bag of Words:
- Ignores the order of words, so “dog bites man” and “man bites dog” are represented identically.
- Does not capture the context or semantic meaning of words.
- Cannot distinguish between synonyms or polysemy (words with multiple meanings).
- The resulting feature vectors are often very high-dimensional and sparse.

---

## Word2Vec and Semantic Embeddings
- To address BoW’s limitations, neural embedding models like Word2Vec were developed.
- Word2Vec learns dense vector representations (embeddings) for words such that words with similar meanings are close together in the vector space.
- It captures semantic relationships (e.g., “king” - “man” + “woman” ≈ “queen”) and context, enabling better performance in downstream NLP tasks.
- Word2Vec uses two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.

---

### Summary Table

| Model | Captures Word Order | Captures Semantics | Output Dimensionality | Example Use Cases |
|--------------|--------------------|--------------------|-----------------------|-------------------------|
| Bag of Words | No | No | High (sparse) | Text classification, IR |
| Word2Vec | No (context window)| Yes | Low (dense) | Similarity, clustering |

---

**Further Reading**
- Explore GloVe, FastText, and contextual embeddings (BERT, GPT) for more advanced semantic understanding.

---

## Tokenization and Data Storage

When processing large text datasets for machine learning, tokenization (splitting text into smaller units like words or subwords) can result in very large data representations. Storing all tokenized data in memory is often impractical due to size constraints.

**Efficient Data Handling:**
- To manage this, the tokenized data is typically serialized and saved to disk in binary formats, such as `training.bin` and `validation.bin` files.
- `training.bin` contains the tokenized data used for training the model, while `validation.bin` holds a separate portion for evaluating model performance during training.
- This approach allows for efficient loading and processing of data in batches, reducing memory usage and enabling the handling of large datasets.

**Summary:**
- Tokenization increases data size significantly.
- Binary files (`training.bin`, `validation.bin`) are used to store and efficiently access large tokenized datasets during model training and validation.

---

## Minimum VRAM Requirement for Embeddings

When working with word embeddings (such as those used in Word2Vec or neural language models), the memory required to store the embedding matrix is a key consideration, especially for GPU/VRAM usage.

**Formula:**
The minimum VRAM required to store the embedding matrix is:

VRAM (bytes) = vocab_size × embedding_dim × bytes_per_value

Where:
- `vocab_size` is the number of unique tokens (words or subwords) in the vocabulary.
- `embedding_dim` (dimension_size) is the size of each embedding vector (e.g., 300 for Word2Vec, 768 for BERT).
- `bytes_per_value` depends on the data type (e.g., 4 bytes for float32, 2 bytes for float16).

**Example:**
If vocab_size = 50,000, embedding_dim = 300, and float32 (4 bytes):

VRAM = 50,000 × 300 × 4 = 60,000,000 bytes ≈ 57.2 MB

**Note:**
- This calculation only covers the embedding matrix. Actual VRAM usage will be higher due to model parameters, activations, gradients, and other overhead during training or inference.

---

## Dimensions in Embeddings

In the context of embeddings and machine learning, "dimensions" refer to the number of characteristics or features used to represent each token (word, subword, etc.) as a vector.

- For example, a 300-dimensional embedding means each word is represented by a vector with 300 numerical values, where each value captures a different aspect or feature of the word’s meaning or usage.

**Summary:**
- Dimensions = features or characteristics encoded in each embedding vector.
---

## Context Window in Language Models

The context window refers to the number of tokens (words or subwords) that a language model considers as input when predicting the next token in a sequence.

- For example, if the context window is 512, the model can use up to 512 previous tokens to generate or predict the next token.
- A larger context window allows the model to capture longer-range dependencies and more context, but also increases computational and memory requirements.

**Summary:**
- The context window defines how many tokens the model "sees" at once to make its next prediction.

---

## Decoder-Only Models in Modern NLP

In recent NLP architectures, such as GPT and other large language models, the traditional encoder is no longer always required. Decoder-only models can perform both understanding (encoding) and generation (decoding) tasks within a single architecture.

- These models process input sequences and generate outputs using only the decoder stack, leveraging self-attention to capture context and meaning.
- The "encoding" of the input is implicitly handled by the same layers that generate the output, making the architecture simpler and more unified.
- This approach is especially common in large language models used for text generation, completion, and instruction following.

**Summary:**
- Modern decoder-only models can handle both input understanding and output generation, eliminating the need for a separate encoder component.
---

## Encoder and Decoder Models

In sequence-to-sequence (seq2seq) architectures, such as those used for machine translation, summarization, and other NLP tasks, the encoder and decoder play distinct but complementary roles:

**Encoder:**
- The encoder model reads and processes the input sequence (e.g., a sentence in natural language).
- It converts the input into a fixed-size context vector (or a sequence of vectors), capturing the meaning and important features of the input.
- The encoder does not generate output in the target language; its job is to understand and represent the input.

**Decoder:**
- The decoder model takes the context vector(s) produced by the encoder and generates the output sequence (e.g., a translated sentence).
- It produces the output one token at a time, using both the context from the encoder and its own previously generated tokens.
- The decoder is responsible for producing the final result in the target language or format.

**Example: Machine Translation**
- Encoder: Reads an English sentence and encodes its meaning.
- Decoder: Takes the encoded meaning and generates the corresponding sentence in French.

**Note:**
- Modern architectures like Transformers use stacks of encoders and decoders, with attention mechanisms to improve context handling.

---

## RMSNorm (Root Mean Square Layer Normalization)

**What_is_RMSNorm?**
- RMSNorm stands for Root Mean Square Layer Normalization.
- It is a normalization technique used in deep learning models, especially in transformer architectures.
- RMSNorm normalizes the input vector by its root mean square (RMS) value, helping to stabilize and accelerate training.


**Formula:**
Given an input vector $x$ of length $d$:

$$
ext{RMSNorm}(x) = x \cdot \frac{g}{\text{RMS}(x)} \\
ext{where}\quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}
$$

Here, $g$ is a learned scaling parameter.

**When & Where is it Applied?**
- RMSNorm is typically applied after linear transformations (such as attention or feed-forward layers) in transformer-based models.
- It is used as an alternative to LayerNorm, offering similar benefits but with slightly lower computational cost and sometimes improved performance.

**Is RMSNorm an Activation Function?**
- No, RMSNorm is not an activation function.
- It is a normalization layer, meaning it adjusts the scale of the input but does not introduce non-linearity like activation functions (e.g., ReLU, GELU).

**Summary:**
- RMSNorm is a normalization technique, not an activation function, and is used to stabilize training in deep learning models, especially transformers.

---

## Why and When Are Scalar Vectors Used?

**What are Scalar Vectors?**
- In the context of normalization layers (like RMSNorm, LayerNorm), a "scalar vector" usually refers to a learned scaling parameter (often denoted as $g$ or $\gamma$) that is multiplied element-wise with the normalized output.

**Why Are They Used?**
- After normalization, the scale and distribution of the data are changed. The learned scalar vector allows the model to restore or adjust the scale of each feature as needed for optimal performance.
- This flexibility helps the model learn more expressive representations and prevents the normalization from being too restrictive.

**When Are They Used?**
- Scalar vectors are used in most normalization layers (BatchNorm, LayerNorm, RMSNorm, etc.) and are learned during training.
- They are applied after the normalization step, before passing the output to the next layer or activation function.

**Do Scalar Vectors Prevent Vanishing Gradients?**
- While normalization layers like RMSNorm help stabilize training, vanishing gradients can still occur in deep networks.
- The learned scalar vectors (scaling parameters) help by adaptively rescaling the normalized outputs, which can mitigate—but not completely eliminate—the vanishing gradient problem.
- By maintaining appropriate activation magnitudes, scalar vectors support better gradient flow through the network, reducing the risk of gradients becoming too small during backpropagation.

**Summary:**
- Scalar vectors in normalization layers are important for training stability, model expressiveness, and can help reduce vanishing gradients, but may not fully eliminate the problem in very deep networks.

---

## MHA, GQA, and MQA in Transformers

**MHA (Multi-Head Attention):**
- Standard in transformer models, MHA uses multiple sets of queries, keys, and values (heads) to capture different aspects of the input sequence.
- Each head learns different attention patterns, and their outputs are concatenated and linearly transformed.

**MQA (Multi-Query Attention):**
- In MQA, each head has its own query, but all heads share the same keys and values.
- This reduces memory and computation costs, making it efficient for large-scale models and inference.

**GQA (Grouped-Query Attention):**
- GQA is a middle ground between MHA and MQA.
- Queries are divided into groups, with each group sharing keys and values, but different groups can have different keys/values.
- This approach balances expressiveness and efficiency, reducing resource usage while retaining more modeling power than MQA.

**Why is GQA More Widely Used?**
- GQA offers a good trade-off between the flexibility of MHA and the efficiency of MQA.
- It is especially popular in recent large language models (e.g., Llama 2/3) because it scales well and maintains strong performance.

**Summary Table:**

| Method | Query | Key | Value | Efficiency | Expressiveness |
|--------|-------|-----|-------|------------|---------------|
| MHA | Many | Many| Many | Low | High |
| GQA | Many | Grouped | Grouped | Medium | Medium-High |
| MQA | Many | One | One | High | Medium |

**References:**
- See Llama 2/3 and other recent transformer architectures for practical use of GQA.

---

## Self-Attention, Causal Attention, and GQA: When and Where Used

**Self-Attention:**
- Used in most transformer-based models (e.g., BERT, encoder layers of sequence-to-sequence models).
- Each token attends to all other tokens in the input sequence, allowing the model to capture global dependencies and context.
- Common in tasks where the entire input is available (e.g., classification, translation, summarization).

**Causal (Masked) Attention:**
- Used in autoregressive models (e.g., GPT, decoder layers in sequence-to-sequence models).
- Each token can only attend to previous tokens (not future ones), ensuring predictions are made one step at a time.
- Essential for text generation, language modeling, and any task where the model must not "see the future."

**GQA (Grouped-Query Attention):**
- GQA is a variant of attention (often combined with self- or causal attention) to improve efficiency and scalability.
- Used in large language models (e.g., Llama 2/3) to reduce memory and computation while maintaining strong performance.
- GQA can be applied in both self-attention and causal attention settings, depending on the model architecture.

**Summary Table:**

| Mechanism | Where Used | Purpose/Scenario |
|-------------------|-----------------------------------|---------------------------------------|
| Self-Attention | Encoders, BERT, full-sequence | Classification, translation, context |
| Causal Attention | Decoders, GPT, generation models | Language modeling, text generation |
| GQA | Large LLMs (Llama 2/3, etc.) | Efficient attention in large models |

**Note:**
- Modern models may combine these mechanisms for optimal performance and efficiency.

---

## Why Are Multi-Heads Used in Attention?
**Example:**

Suppose the input sentence is: "The cat sat on the mat."

- One attention head might focus on the relationship between "cat" and "sat" (subject-verb).
- Another head might focus on "sat" and "mat" (verb-object).
- Yet another head might focus on the position of "the" (articles) relative to nouns.

By combining these different perspectives, the model builds a richer understanding of the sentence than any single head could provide.

**Purpose of Multi-Head Attention:**
- Multi-head attention allows the model to focus on different parts of the input sequence simultaneously.
- Each head learns to capture different types of relationships, patterns, or dependencies between tokens.

**Benefits:**
- Increases the model’s ability to represent complex information by combining multiple perspectives.
- Helps the model learn richer, more diverse features from the data.
- Reduces the risk of missing important context that might be overlooked by a single attention mechanism.

**How It Works:**
- The input is projected into multiple sets of Q, K, V vectors (one set per head).
- Each head performs attention independently, and their outputs are concatenated and linearly transformed to produce the final result.

**Summary:**
- Multi-head attention enhances the model’s expressiveness and ability to capture complex relationships in the data.

---

## Sliding Window Attention: When Is It Used?

**What Is Sliding Window Attention?**
- Sliding window attention is a variant of attention where each token attends only to a fixed-size window of neighboring tokens, rather than the entire sequence.

**When and Why Is It Used?**
- Used in long-sequence models (e.g., Longformer, BigBird) to reduce memory and computation costs.
- Essential when the input sequence is too long for standard self-attention (which scales quadratically with sequence length).
- Common in document processing, long text analysis, genomics, and any task with very long input sequences.

**How Does It Work?**
- Each token attends only to tokens within a local window (e.g., 256 tokens before and after), not the whole sequence.
- This makes attention computation linear with sequence length, enabling efficient processing of long contexts.

**Summary:**
- Sliding window attention is used for efficient modeling of long sequences, trading off some global context for scalability.
---


---
### How QKV Works in Attention

- In transformer models, each input (e.g., a word or token) is projected into three vectors:
- **Query (Q)**
- **Key (K)**
- **Value (V)**
- These vectors are used to compute attention scores, which determine how much focus each token should give to others in the sequence.


$$
ext{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Where:
- $Q$ = Query matrix
- $K$ = Key matrix
- $V$ = Value matrix
- $d_k$ = Dimension of the key vectors (for scaling)
- $QK^T$ computes the similarity between queries and keys
- softmax normalizes the scores across each row
- The result is a weighted sum of values for each query



---
**Analogy with Employee Table:**

| **EmployeeID** | **Name** |
|:--------------:|:-----------:|
| 101 | Alice |
| 102 | Bob |
| 103 | Charlie |

Imagine you want to find information about an employee (e.g., "Who is EmployeeID 102?"):
- Your search term ("102") is like the **Query**.
- The EmployeeID column acts as the **Key**.
- The Name column acts as the **Value**.

**The process:**
1. Compare your Query (102) with all Keys (EmployeeIDs) to find a match.
2. When a match is found, retrieve the corresponding Value (Name: Bob).


---
**In Transformers:**
- Each token’s **Query** is compared (via dot product) with all **Keys** to compute attention weights.
- These weights are used to combine the **Values**, producing a context-aware representation for each token.

**Summary:**
- **Query:** What you’re looking for (e.g., EmployeeID 102)
- **Key:** What you compare against (e.g., all EmployeeIDs)
- **Value:** What you retrieve (e.g., Name)
---
---

## KV Cache (Key-Value Cache)

**Formula to Calculate KV Cache Size (in bytes):**

$$
ext{KV Cache Size (bytes)} = N_{layers} \times N_{heads} \times L_{context} \times D_{head} \times 2 \times S_{bytes}
$$

Where:
- $N_{layers}$ = Number of transformer layers in the model
- $N_{heads}$ = Number of attention heads per layer
- $L_{context}$ = Context window size (number of tokens to cache)
- $D_{head}$ = Dimension of each attention head
- $2$ = For both Key and Value tensors (each stored separately)
- $S_{bytes}$ = Size in bytes of each value (e.g., 2 for float16, 4 for float32)

**Explanation:**
- For each layer and each head, the model stores both Key and Value tensors for every token in the context window.
- The total size is the product of all these factors, giving the memory required for the KV cache during inference.

**What is KV Cache?**
- KV cache stands for Key-Value cache, a technique used in transformer-based models during inference (especially in autoregressive generation).
- It stores the key and value tensors computed at each layer for previously processed tokens.

**Why is KV Cache Used?**
- In autoregressive generation (e.g., text generation with GPT), the model generates one token at a time.
- Without caching, the model would recompute keys and values for all previous tokens at every step, which is inefficient.
- KV cache allows the model to reuse previously computed keys and values, so only the new token’s keys and values need to be computed at each step.
- This greatly speeds up inference and reduces memory usage.

**When and Where is KV Cache Applied?**
- KV cache is used during inference (not training) in autoregressive transformer models (e.g., GPT, Llama, etc.).
- It is essential for efficient text generation, chatbots, and any application where the model generates sequences token by token.

## Reference links
- https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
- https://www.deeplearning.ai/short-courses/pretraining-llms/
- https://github.com/ideaweaver-ai/building-gemma4-from-scratch/blob/dev/build_tokenizer.py
- https://huggingface.co/spaces/lakhera2023/gemma4-nano-tinystories-demo
- https://console.runpod.io/deploy?type=GPU
- https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#why-the-question-nobody-wants-to-answer
- https://www.ideaweaver.ai/l/dashboard
  
