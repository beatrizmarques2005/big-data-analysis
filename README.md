#  Big Data Analysis with PySpark  

*This README provides an overview of the repository structure. Since only a powerpoint presentation was required for this assignment, the README serves as a brief guide to explain the purpose and contents of each folder.*


## Quick Summary  

| Project                     | Focus Area                     | Key Spark Component | ML Task Type     |
|----------------------------|--------------------------------|---------------------|------------------|
| **Books Network**          | Network/Graph Data             | GraphFrames         | Graph Analytics  |
| **Portuguese Bank Marketing** | Financial & Marketing Data     | Spark MLlib        | Binary Classification   |

---

## Repository Structure  

```tree
big-data-analysis/
│
├── data/
│   ├── bank_split_ids/                 # Train/Val/Test ID lists for stratified partitioning
│   │   ├── test_ids/
│   │   ├── train_ids/
│   │   └── val_ids/
│   └── raw/
│       └── BankMarketing.csv           # Original dataset
│
├── notebooks/
│   ├── BankMarketing/
│   │   ├── 01_exploration.ipynb    
│   │   ├── 02_preprocessing.ipynb     
│   │   ├── 03_modelling.ipynb    
│   │   └── 04_deployment.ipynb     
│   │
│   ├── Books/                    
│   │   ├── 01_exploration_and_preprocessing.ipynb
│   │   ├── 02_GraphFrames.ipynb
│   │   ├── 04_graphframes.ipynb
│   │   └── 05_summary_visuals.ipynb
│   │
│   └── MongoDB_connection.ipynb        # MongoDB import and validation
│
├── source/
│   ├── pipelines/
│   │   ├── bank_preproc_pipeline_model     # Saved BankMarketing preprocessing fitted pipeline
│   │   └── bank_preproc_pipeline_model_u   # Saved BankMarketing preprocessing unfitted pipeline
│   │
│   ├── ml_functions.py                 # ML models and evaluators
│   ├── preprocessing.py                # Custom transformers & data preparation
│   └── visualizations.py               # Plotting and Spark visualization helpers
│
├── spark_checkpoints/                  # Metadata for structured Spark checkpoints
│
├── .gitignore
└── README.md
```

---

## Team  

- Beatriz Marques – 20231605  
- David Carrilho – 20231693  
- Duarte Fernandes – 20231619  
- Filipe Caçador – 20231707  
- Mariana Calais-Pedro – 20231641  
