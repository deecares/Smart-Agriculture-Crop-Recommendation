# Smart Agriculture Data Analytics using Apache Spark

A Big Data Analytics mini-project using **Apache Spark (PySpark, Spark SQL and Spark MLlib)** to analyse district-wise crop production statistics across India.

## 1. Problem statement

Agricultural production varies by state, district, crop, season and year. This project processes historical crop production records to identify production patterns and estimate crop production using cultivated area and contextual features.

- **Selected case study:** 11.4 – Smart Agriculture Data Analytics
- **Primary technology:** Apache Spark / PySpark
- **ML problem type:** Regression
- **Execution:** Google Colab with Spark local mode (not a multi-node cluster)

## 2. Dataset

- **Name:** District-wise, Season-wise Crop Production Statistics (1997–2015)
- **Original source:** Ministry of Agriculture & Farmers Welfare, Government of India, Directorate of Economics & Statistics
- **Source platform:** Open Government Data (OGD), data.gov.in
- **Official resource:** https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997
- **Public mirror used by notebook:** https://github.com/JitendraAmbekar/india-crop-production-analysis
- **Records:** approximately 246,091 raw records
- **Coverage:** 1997–2015; about 33 states/UTs, 646 districts and 124 crop types
- **Fields:** State_Name, District_Name, Crop_Year, Season, Crop, Area, Production

The Colab notebook downloads the CSV from a public GitHub mirror because the official government portal may block automated requests. The original source remains the Government of India dataset. The mirror's byte-for-byte identity with the official download has not been independently verified.

## 3. Big Data processing and analytics

The PySpark program:
1. Reads the CSV with Spark DataFrames.
2. Converts numeric fields, trims categorical values, removes incomplete/invalid rows and duplicates.
3. Excludes Coconut from comparable crop totals and model training because its reported production units are inconsistent with the other crops in the source.
4. Calculates top crops by total production and area.
5. Calculates state-wise and season-wise production.
6. Produces yearly production and cultivated-area trends.
7. Saves analytical results as CSV output folders.

## 4. Machine-learning task

**Target:** Production.

**Features:** Area, year, state, district, crop and season. The derived Yield field is not used, avoiding target leakage because yield is calculated from production and area.

Spark MLlib compares:
- Random Forest Regressor
- Gradient-Boosted Tree Regressor

Evaluation metrics: RMSE and R² on a held-out test split. The best model must be selected from the actual run output; no accuracy or performance result is claimed before execution.

## 5. Run the project

Open the notebook in Google Colab:

https://colab.research.google.com/github/deecares/Smart-Agriculture-Crop-Recommendation/blob/main/notebooks/Run_on_Google_Colab.ipynb

Run cells from top to bottom. The notebook attempts to download the dataset automatically and provides an upload fallback if both mirrors are unavailable.

For a local Spark environment:

```bash
pip install -r requirements.txt
spark-submit src/agriculture_analytics.py --input data/crop_production.csv --output outputs
```

## 6. Repository structure

```text
Smart-Agriculture-Crop-Recommendation/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── Run_on_Google_Colab.ipynb
└── src/
    ├── agriculture_analytics.py
    └── crop_recommendation.py  # earlier small-dataset prototype
```

## 7. Scope and limitations

The records are historical district-level administrative statistics, not farm-level measurements. Production units may not be fully comparable for every crop; Coconut is excluded from this analysis due to a known unit inconsistency. Spark runs in local mode in the supplied Colab notebook. A multi-node cluster deployment is a possible extension. Report actual cleaned row counts and model metrics from the completed notebook execution.
