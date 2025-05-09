# 📝 Arabic Fact-Checking  

This repository is dedicated to **fact-checking Arabic claims**. It includes tools, data, and scripts for retrieving and verifying claims.  

---

## 📁 Repository Structure  

### **🗂 claim_retriever/**  
This directory contains the core scripts and utilities for processing claims:  

| File | Description |
|------|-------------|
| **`api.py`** | Handles API-related functionality for retrieving or serving claim data. |
| **`config.py`** | Contains configuration settings for the project. |
| **`main.py`** | The main entry point for running the claim retrieval script. |
| **`queries.txt`** | Sample queries or search terms for testing. |
| **`utils.py`** | Utility functions used across the project. |

🔄 **Results** are saved in the output file **`claim_reviews.json`**.  

---

## 🚀 Getting Started  

1. Clone the repository:  
    ```bash
    git clone https://github.com/your-username/Arabic-Fact-Checking.git
    cd Arabic-Fact-Checking/claim_retriever
    ```

<!-- 2. Install the required packages:  
    ```bash
    pip install -r requirements.txt
    ``` -->

2. Add the `.env` variables

3. Run the main script:  
    ```bash
    python main.py
    ```
