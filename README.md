# Smart Agriculture Data Analytics using Apache Spark

A Big Data Analytics mini-project using **Apache Spark (PySpark and Spark SQL)** to process district-level agricultural statistics for India. The project performs large-scale data cleaning, aggregation and trend analysis, then uses Spark MLlib regression to estimate crop production.

## 1. Problem statement

Agricultural production varies across states, districts, crops, seasons and years. This project analyses historical Area, Production and Yield (APY) records to identify production patterns, compare regional and seasonal performance, and estimate production from cultivated area and contextual features.

**Selected mini case study:** 11.4 – Smart Agriculture Data Analytics Platform  
**Primary BDA technology:** Apache Spark / PySpark / Spark SQL  
**ML problem type:** Regression  
**Execution environment:** Google Colab with Spark local mode (not a multi-node cluster)

## 2. Dataset

- **Dataset:** Crop Wise Area Production Yield (APY)
- **Source platform:** India Data Portal
- **Original source:** Ministry of Agriculture & Farmers Welfare, Government of India
- **Dataset page:** https://indiadataportal.com/p/area-production-yield-apy
- **Listed download size:** approximately 53.62 MiB
- **Coverage:** 1997–2023
- **Geographic granularity:** district level; 34 States/UTs and 740 districts reported in the portal metadata
- **Main fields:** Year, State Name, District Name, Crop Name, Crop Code, Crop Type, Season, Area, Production and Yield
- **License shown by the portal:** Open Data Commons Attribution License

Download the CSV from the dataset page and upload it to the Colab notebook when prompted. The notebook saves it as `data/APY.csv`. Do not commit the raw dataset to GitHub; it is large and is downloaded separately for execution.

## 3. Big Data processing and analytics

The PySpark program:
1. Reads the large CSV with Spark DataFrames.
2. Selects required columns, converts numeric fields, removes incomplete/invalid records and extracts the starting year.
3. Computes top crops by total production.
4. Computes state-wise production and cultivated area.
5. Compares production and average yield by season.
6. Produces year-wise production and area trends.
7. Saves each analytical result as CSV output folders.

## 4. Machine-learning task

**Regression target:** Production.

Input features include cultivated area, year, state, district, crop name/code, crop type and season. The pipeline uses StringIndexer, OneHotEncoder and VectorAssembler in Spark MLlib, then compares:
- Random Forest Regressor
- Gradient-Boosted Tree Regressor

Models are evaluated using RMSE and R² on a held-out test split. The best model must be selected from the actual run output; no result is claimed before execution.

## 5. Run the project

Open the notebook in Google Colab:

https://colab.research.google.com/github/deecares/Smart-Agriculture-Crop-Recommendation/blob/main/notebooks/Run_on_Google_Colab.ipynb

Run cells from top to bottom. When prompted, upload the downloaded APY CSV. The notebook runs the Spark script and displays the model comparison. Analytical outputs are saved under `outputs/`.

For a local Spark environment:

```bash
pip install -r requirements.txt
spark-submit src/agriculture_analytics.py --input data/APY.csv --output outputs
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

The dataset is historical district-level administrative agricultural statistics. Results describe recorded patterns and should not be treated as farm-level advice. The Colab notebook runs Spark in local mode; a multi-node Hadoop/Spark cluster deployment is a possible extension. Actual row counts, cleaning counts and model metrics should be taken from the completed execution output.
