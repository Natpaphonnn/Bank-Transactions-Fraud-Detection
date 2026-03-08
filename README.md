# Bank Transactions Fraud Detection

Fraud Detection on Synthetic Bank Transactions — EDA, Feature Engineering, Anomaly Detection (Isolation Forest, DBSCAN, Autoencoder)

## Live Presentation

https://natpaphonnn.github.io/Bank-Transactions-Fraud-Detection/presentation.html

## Project Structure

| File | Description |
|------|-------------|
| `01_EDA_Bank_Transactions.ipynb` | Exploratory Data Analysis — data overview, distributions, anomaly indicators |
| `02_Feature_Engineering_and_Anomaly_Detection.ipynb` | Feature Engineering (~30 features) + 3 Anomaly Detection models |
| `03_Model_Evaluation_and_Comparison.ipynb` | Model evaluation, comparison, sensitivity analysis & business insights |
| `presentation.html` | Interactive presentation website (banking theme) |
| `bank_transactions.csv` | Raw dataset — 50,000 synthetic bank transactions |

## Key Findings

- **50,000 transactions** from **495 accounts** across **43 locations** (2020–2025)
- **4.75%** of transactions are amount outliers (above $899)
- **609 devices** shared across multiple accounts (max 9 accounts per device)
- **3.85%** of transactions have 3+ login attempts (potential brute force)
- **266 accounts** active in 5+ different cities

## Models Used

| Model | Approach |
|-------|----------|
| Isolation Forest | Tree-based isolation of outliers |
| DBSCAN | Density-based clustering (noise = anomaly) |
| Autoencoder / LOF | Reconstruction error / Local density |

Results are combined into a **Composite Risk Score** (Low / Medium / High / Critical).

## Tech Stack

Python, Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, Chart.js
