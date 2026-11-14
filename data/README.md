# Dataset instructions

This project uses the **Crop Wise Area Production Yield (APY)** dataset.

- Dataset page: https://indiadataportal.com/p/area-production-yield-apy
- Source: Ministry of Agriculture & Farmers Welfare, Government of India
- Download format: CSV
- Coverage: 1997–2023
- Geographic granularity: district level
- Portal-listed download size: approximately 53.62 MiB

Download the CSV using the **Download Data** option on the dataset page. In Google Colab, the notebook prompts you to upload the downloaded file and saves it as:

`data/APY.csv`

Expected columns include:

`Year, State Name, State Code, District Name, District Code, Crop Name, Crop Code, Crop Type, Season, Area, Area Unit, Production, Production Unit, Yield, Yield Unit`

Do not upload the raw dataset to this GitHub repository; the Colab notebook accepts it separately.
