# 🌍 Graph RAG Travel Assistant ✈️

Welcome to **Graph RAG Travel Assistant**, a cutting-edge Retrieval-Augmented Generation (RAG) system designed to revolutionize how you plan your travels. 🚀 This system combines a knowledge graph, state-of-the-art embeddings, and large language models (LLMs) to provide personalized travel guidance.

---

## 📜 **How It Works (Pipeline Overview)**

The Graph RAG system operates in **three primary layers**, each designed to enhance user interactions and deliver optimal results:

### 1️⃣ **Input Preprocessing Layer**
This is where the magic begins:
- **Intent Classification**: Understand user queries like "Find hotels near a beach" or "What places can I visit without a visa?".
- **Entity Extraction**: Detect key elements like cities, dates, or traveler types.
- **Embedding Generation**: Convert user inputs into vector representations for efficient semantic retrieval.

*Featured in*: [`01_input_preprocessing.py`](https://github.com/helguindy/Graph-RAG-Travel-Assistant/blob/main/01_input_preprocessing.py)

---

### 2️⃣ **Graph Retrieval Layer**
Leverage a **Neo4j-powered Knowledge Graph** to generate precise results:
- **Baseline Retrieval**: Use Cypher queries for deterministic, structured data retrieval.
- **Embedding Retrieval**: Apply semantic similarity to locate contextual knowledge across various entries.
- Combines both approaches for **hybrid retrieval.**

*Explore more*: [`02_graph_retrieval.py`](https://github.com/helguindy/Graph-RAG-Travel-Assistant/blob/main/02_graph_retrieval.py)

---

### 3️⃣ **LLM Response Generation Layer**
Produce natural, insightful language responses:
- Merge structured data retrieval results into meaningful contexts.
- Supports multiple LLMs, such as OpenAI, HuggingFace, and Anthropic models.
- Evaluates models to determine the best one for your needs (e.g., speed vs. accuracy).

*Dive deeper*: [`03_llm_layer.py`](https://github.com/helguindy/Graph-RAG-Travel-Assistant/blob/main/03_llm_layer.py)

---

## 🛠️ **Key Features**
### 🎯 **Custom Query Handling**
Ask anything, ranging from:
- **Visa requirements**: "Where can I travel without a visa?"
- **Recommendations**: "Best beaches near Sydney?"
- **Local Attractions**: "Things to do in Paris for couples?"

### ⚙️ **Multi-Model Support**
Choose models that match your requirements:
- **Gemma 2B (Fast)** ✅
- **Mistral 7B (Versatile)** 🌟 
- **Llama 3.2 1B (Cost-effective)** 🌍

### 🌐 **Knowledge Graph Visualization**
Two UIs:
1. **Architecture UI**: Friendly for devs and researchers to explore the pipeline visually. (Uses PyVis for graph rendering.)
2. **Chat UI**: A polished, ChatGPT-like interface for end-users.

Want a closer look? Check: [`architectureUI.py`](https://github.com/helguindy/Graph-RAG-Travel-Assistant/blob/main/architectureUI.py) | [`app.py`](https://github.com/helguindy/Graph-RAG-Travel-Assistant/blob/main/app.py)

---

## ✨ **Why Use Graph RAG Travel Assistant?**
1. 📊 **High Accuracy**: Combines both retrieval and generation for precise results.
2. 🌟 **Personalization**: Adjust plans based on your travel profile.
3. 💻 **Developer-Friendly**: Easily expandable and highly modular.

---

Feel free to explore or contribute to this open-source project. Let's build smarter travel tools, *together*! 🛫