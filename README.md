# Arabic Fact-Checking 📰

This repository provides tools to **retrieve evidence, generate QA pairs, and verify claims** using large language models (LLMs). It is designed for researchers and developers who want to experiment with fact-checking pipelines, retrieval, and evaluation.


## ⚡ Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/arabic-fact-checking.git
cd arabic-fact-checking
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline (end-to-end)
```bash
python src/scripts/run_all.py
```

This will:

1. Retrieve evidence
2. Generate QA pairs
3. Predict claim labels
4. Evaluate results

---

### 🛠️ Usage

🔹 Evidence Retrieval

Retrieve supporting text for claims:
```bash
python retrieval/claim_retriever/main.py
```

🔹 QA Pair Generation

Generate question–answer pairs for fact-checking:
```bash
# From gold evidence
python src/scripts/generate_gold_qa_pairs.py  

# From retrieved evidence
python src/scripts/generate_retrieved_evidence_qa_pairs.py
```

### 🔹 Label Prediction

Predict claim labels (`True`, `False`, `Unsupported`):

```bash
python src/scripts/predict_labels.py
```

---

### 🤝 Contributing

Contributions are welcome! If you’d like to improve retrieval methods, add new evaluation metrics, or extend the verification pipeline, feel free to fork the repo and submit a pull request.

