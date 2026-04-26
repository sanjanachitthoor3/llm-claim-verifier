# LLM Claim Verification System

An **explainable AI system** that detects hallucinations in LLM outputs by verifying individual claims against real-world evidence.

---

## 🚀 What it does

Given an LLM response, the system:

- Extracts factual claims
- Retrieves relevant information from Wikipedia
- Finds semantically similar evidence using FAISS
- Verifies each claim using an NLI model
- Produces:
  - **Verdict per claim** (Supported / Contradicted / Not Enough Info)
  - **Evidence sentences**
  - **Trust score & hallucination risk**

---

## 🧩 Why this matters

LLMs often generate **confident but incorrect information**.

This system:

- Breaks responses into **verifiable units (claims)**
- Grounds decisions in **retrieved evidence**
- Avoids blind trust by returning **Not Enough Info when uncertain**

---

## 🧠 Explainability (Key Feature)

Unlike black-box hallucination detectors, this system is **fully interpretable**:

- Every claim is paired with **retrieved evidence**
- Final verdicts are **traceable to source text**
- Users can see _why_ something is marked correct or incorrect

> This makes the system suitable for **trust-critical applications**.

---

## ⚙️ Pipeline

```id="pipeline"
LLM Response
   ↓
Claim Extraction
   ↓
Wikipedia Retrieval
   ↓
FAISS Semantic Search (Top-K Evidence)
   ↓
NLI Verification
   ↓
Explainable Output + Trust Score
```

---

## ✨ Core Features

- Claim-level verification (fine-grained)
- Evidence-backed reasoning
- FAISS-based semantic retrieval
- Transformer-based NLI verification
- Quantitative trust scoring
- Transparent, explainable outputs

---

## 🛠️ Tech Stack

- FastAPI
- Sentence Transformers
- Hugging Face Transformers
- FAISS
- Wikipedia API
- HTML/CSS/JS

---

## 📦 Run Locally

```bash
git clone https://github.com/your-username/llm-claim-verifier.git
cd llm-claim-verifier

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

uvicorn app.api.main:app --reload
```

Open:

```id="run"
frontend/index.html
```

---

## 📸 Demo

### Mixed Claims (Realistic LLM Output)

![Mixed](demo-images\image-3.png)
![Mixed](demo-images\image-4.png)
![Mixed](demo-images\image-5.png)

### Fully Supported Case

![Supported](demo-images\image-6.png)
![Supported](demo-images\image-7.png)

---

## ⚠️ Limitations

- Single-source verification (Wikipedia only)
- Dependent on retrieval quality (missed evidence → Not Enough Info)
- No multi-hop reasoning
- Moderate compute requirements

---

## 🔮 Future Work

- Multi-source verification
- Hybrid retrieval (keyword + semantic)
- Lightweight models for deployment
- Evidence highlighting at token level

---

## 👤 Author

Sanjana Chitthoor
NLP Project – LLM Reliability & Hallucination Detection
