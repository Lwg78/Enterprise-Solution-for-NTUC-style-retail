# 🛒 Retail Sales Forecasting Pipeline (Armstrong Cycle Implementation)

## 📌 Executive Summary
This project implements an end-to-end Machine Learning pipeline to predict daily retail sales. Drawing from experience in the FMCG/Supermarket sector (NTUC FairPrice), it addresses the challenge of forecasting sales trends in high-volume, seasonal environments using the **Martin Armstrong Economic Confidence Model (ECM)** coupled with modern Ensemble Learning (Random Forest).

## 🛠 Tech Stack
* **Language:** Python 3.9+
* **Data Engineering:** Pandas (ETL), Glob (File Pattern Matching)
* **Machine Learning:** Scikit-Learn (Random Forest, Custom Transformers)
* **Architecture:** Modular Pipeline (Clean -> Preprocess -> Feature Engineering -> Model)

## 🚀 Key Features
* **Automated Data Ingestion:** A "Universal Scraper" capable of ingesting and merging disparate Excel reports (`.xls`, `.xlsx`) with varying header formats.
* **Custom Feature Engineering:** A custom Scikit-Learn Transformer (`ArmstrongCycleTransformer`) that mathematically models long-term economic cycles (8.6 years) and short-term business seasonality (365 days).
* **Simulation Mode:** Includes a synthetic data generator to validate model performance on long-term trends when historical data is sparse.

## 📂 Project Structure
```text
├── data/               # Raw Excel files & Processed CSVs
├── src/
│   ├── data_cleaner.py # Robust header detection & merging
│   ├── ArmstrongCycleTransformer.py # The core math logic
│   ├── preprocessor.py # Aggregation & Time-series formatting
│   └── datagen.py      # Synthetic data simulation
├── main.py             # The Commander script
└── README.md
```

## 🧠 Business Logic & Domain Knowledge
* **Problem:** Retail data is often messy ("dirty" headers, multiple Excel tabs).

* **Solution:** The clean_and_merge_data module uses intelligent scanning to find the true header row, ensuring 0% data loss during ingestion.

* **Application:** Useful for Category Managers needing to forecast "Grocery", "Fresh", and "HABA" department sales.