# Dataset

**District-wise, Season-wise Crop Production Statistics (1997–2015)**

- Original source: Ministry of Agriculture & Farmers Welfare, Government of India, Directorate of Economics & Statistics.
- Official resource: https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997
- Public mirror: https://github.com/JitendraAmbekar/india-crop-production-analysis
- Expected raw records: approximately 246,091.
- Columns: State_Name, District_Name, Crop_Year, Season, Crop, Area, Production.

The Colab notebook downloads the CSV as `data/crop_production.csv`. It tries two public raw GitHub mirrors and then offers an upload fallback.

Do not commit the raw CSV to this repository. The original government source is cited above; the mirror's byte-identity with the official file has not been independently verified.
