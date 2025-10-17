# ⚡ Big Data Analysis with PySpark  

Exploration and analysis of up to three large-scale datasets using **PySpark** to demonstrate big data processing, data engineering, machine learning, and analytical capabilities.

---

## 🎯 Project Objective  

This project simulates a **real-world big data consulting scenario**, where our group acts as data scientists at *BigDataCompany*.  
The goal is to demonstrate technical and analytical proficiency using **PySpark** through:  

- **ETL & Data Cleaning** using RDDs and DataFrames  
- **Exploratory Data Analysis** and summarization  
- **Machine Learning Pipelines** with Spark MLlib  
- **Graph Analytics** using GraphFrames  
- *(Optional Bonus)*: Integration with **MongoDB** and **Streaming**  

Each dataset (up to three) highlights different Spark functionalities — such as classification, forecasting, or network analysis — to showcase a comprehensive understanding of big data processing.

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

## 🧩 Methodology  

The workflow follows an **iterative data science process** inspired by **CRISP-DM**, adapted to a big data context:  

1. **Data Understanding** – Explore and describe datasets.  
2. **Data Preparation** – Build ETL pipelines using PySpark.  
3. **Modeling** – Implement ML pipelines with Spark MLlib.  
4. **Evaluation** – Assess performance and interpret insights.  
5. **Deployment & Communication** – Present results and scalability considerations.  

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
