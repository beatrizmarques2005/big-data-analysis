# ⚡ Big Data Analysis with PySpark  

Exploration and analysis of up to three large-scale datasets using **PySpark** to demonstrate big data processing, data engineering, machine learning, and analytical capabilities.

---

## 🎯 Project Objective  

This project simulates a **real-world big data consulting scenario**, where our group acts as data scientists at *BigDataCompany*. The goal is to demonstrate technical and analytical proficiency using **PySpark** through:  

- **ETL & Data Cleaning** using RDDs and DataFrames  
- **Exploratory Data Analysis** and summarization  
- **Machine Learning Pipelines** with Spark MLlib  
- **Graph Analytics** using GraphFrames  
- *(Optional Bonus)*: Integration with **MongoDB** and **Streaming**  

Each dataset highlights different Spark functionalities — such as classification, forecasting, or network analysis — to showcase a comprehensive understanding of big data processing.

### 1️⃣ Portuguese Bank Marketing – Predicting Term Deposit Subscription

**Goal**  
Predict whether a client will subscribe to a **term deposit** based on personal, financial, and campaign-related attributes collected during a direct marketing campaign by a Portuguese bank.

**Key Steps**  
- **Data Cleaning & Preprocessing**: Dropping irrelevant data, treating outliers, and handling missing values  
- **Feature Engineering**: Creating derived features from campaign history, contact duration, and loan indicators  
- **Encoding**: Transforming categorical variables using `StringIndexer` and `OneHotEncoder`  
- **Scaling**: Normalizing numerical features to improve model performance  
- **Modeling**: Classification models using **Spark MLlib** (Logistic Regression, Random Forest, Decision Tree)  
- **Evaluation**: Model performance assessed using **Accuracy**, **Precision**, **Recall**, and **F1-score**

**Highlights**  
- Applies **Spark MLlib** for scalable classification tasks  
- Tackles **class imbalance** with stratified sampling and thoughtful preprocessing  
- Demonstrates how marketing and financial data can drive **predictive insights** for customer targeting

---

### 2️⃣ Books Network – Graph Analytics

**Goal:**  
Analyze a **book co-purchasing network** to uncover community structures, central books, and recommendation patterns using **GraphFrames**.

**Key Steps:**  
- **Graph Construction:** Nodes represent books; edges represent co-purchases or similarity links.  
- **Network Metrics:** Calculation of **PageRank**, **in-degree/out-degree**, and **connected components**.  
- **Community Detection:** Identifying clusters of related books using **Label Propagation**.  
- **Visualization:** Graph insights visualized via network plots and summary statistics.

**Highlights:**  
- Demonstrates **GraphFrames** and **network analysis** in PySpark.  
- Provides insights into **book recommendation systems** and **reader communities**.  

---

### 🧠 Summary  

| Project                     | Focus Area                     | Key Spark Component | ML Task Type     |
|----------------------------|--------------------------------|---------------------|------------------|
| **Books Network**          | Network/Graph Data             | GraphFrames         | Graph Analytics  |
| **Portuguese Bank Marketing** | Financial & Marketing Data     | Spark MLlib        | Classification   |

---

## 📁 Repository Structure  

```tree
big-data-analysis/
│
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/               # Cleaned / transformed data
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_ml_pipeline.ipynb
│   ├── 04_graphframes.ipynb
│   ├── 05_summary_visuals.ipynb
│   └── bonus_streaming.ipynb
│
├── src/
│   ├── spark_utils.py           # Spark session builder and helper functions
│   ├── etl_functions.py         # ETL and preprocessing logic
│   ├── ml_functions.py          # ML pipeline components
│   └── graph_functions.py       # Graph analytics
|
└── README.md
```

---

## 👥 Team  

- Beatriz Marques – 20231605  
- David Carrilho – 20231693  
- Duarte Fernandes – 20231619  
- Filipe Caçador – 20231707  
- Mariana Calais-Pedro – 20231641  

---

## ⚙️ Technology Stack  

- **PySpark** (RDDs, DataFrames, SQL, MLlib, GraphFrames)  
- **Python** (for auxiliary functions and utilities)  
- **Lightning AI** (for orchestration and experimentation)  
- **Matplotlib / Plotly / Seaborn** (for visualization)  
- *(Optional)* MongoDB, Spark Streaming  

---

## 🧪 Git Workflow Guide  

### 🔀 Step 1: Check your current branch  
```bash
git branch
```

### 🔄 Step 2: Sync with shared branch  
```bash
git pull origin common-branch
```

### 💾 Step 3: Save and push your changes  
```bash
git add .
git commit -m "Added ML pipeline notebook"
git push origin your-branch-name
```

### 🔁 Step 4: Merge to shared branch  
```bash
git checkout common-branch
git merge your-branch-name
git push origin common-branch
```

---

## 🗓️ Deliverables  

- **All executed notebooks** (`.ipynb`) with outputs visible  
- **Final presentation** (exported as `.pdf` using the official NOVA IMS template)  
- *(Optional)* MongoDB integration or streaming demo for bonus points  
