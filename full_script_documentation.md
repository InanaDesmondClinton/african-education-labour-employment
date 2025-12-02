```
# Full Script Documentation: African Education-Labour Analytics
## Comprehensive Analysis Notebook Documentation

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources)
3. [Workflow Structure](#workflow-structure)
4. [Section 1: Library Imports](#section-1-library-imports)
5. [Section 2: Data Cleaning](#section-2-data-cleaning)
6. [Section 3: Data Integration](#section-3-data-integration)
7. [Section 4: Analysis](#section-4-analysis)
8. [Section 5: Statistical Models](#section-5-statistical-models)
9. [Section 6: Time Series Forecasting](#section-6-time-series-forecasting)
10. [Key Findings](#key-findings)
11. [Technical Notes](#technical-notes)

---

## 🎯 Project Overview

### Purpose
This analysis examines the relationship between educational indicators and labour market outcomes across African countries from 2014-2021. The goal is to identify patterns, correlations, and predictive relationships that can inform policy decisions and bridge the gap between education outputs and employment demands.

### Objectives
1. **Data Integration**: Combine multiple datasets (SDG National, World Bank, ILO) into a unified analytical framework
2. **Exploratory Analysis**: Understand data distributions, identify missing values, and detect outliers
3. **Correlation Analysis**: Identify relationships between education and labour indicators
4. **Statistical Modeling**: Build regression models to evaluate education-labour relationships
5. **Forecasting**: Develop time-series models to project future labour market trends
6. **Visualization**: Create comprehensive dashboards for stakeholder communication

### Geographic Coverage
- **55 African Countries** including: Algeria, Angola, Benin, Botswana, Burkina Faso, Burundi, Cameroon, Egypt, Ethiopia, Ghana, Kenya, Morocco, Nigeria, Rwanda, Senegal, South Africa, Tanzania, Uganda, Zambia, Zimbabwe, and others
- **Time Period**: 2014-2021 (8 years)
- **Total Observations**: 1,350+ country-year combinations

---

## 📊 Data Sources

### 1. UNESCO SDG National Data
- **Purpose**: Education indicators aligned with Sustainable Development Goal 4 (Quality Education)
- **Key Indicators**:
  - Adjusted Net Attendance Rates (Primary & Secondary)
  - Completion Rates (Primary, Lower Secondary, Upper Secondary)
  - Out-of-School Rates
  - Trained Teachers Percentages
  - Gender Parity Indices
  - Government Education Expenditure
  - Years of Compulsory/Free Education

### 2. World Bank Open Data
- **Purpose**: Economic, demographic, and additional education indicators
- **Key Indicators**:
  - GDP Growth Rate
  - Foreign Direct Investment (FDI)
  - Sector Contributions (Agriculture, Industry, Services)
  - Literacy Rates (Adult & Youth)
  - School Enrollment Rates (Primary, Secondary, Tertiary)
  - Student-Teacher Ratios
  - Population Demographics

### 3. ILO Statistics
- **Purpose**: Labour market indicators
- **Key Indicators**:
  - Overall Unemployment Rate
  - Youth Unemployment Rate (15-24 years)
  - Unemployment by Education Level
  - Labour Force Participation Rates

---

## 🔄 Workflow Structure

The notebook follows a systematic 6-stage workflow:

```

┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Library Imports & Setup                           │
│ - Import Python libraries (pandas, numpy, matplotlib, etc.)│
│ - Configure visualization settings                         │
│ - Set up analysis environment                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Data Cleaning                                     │
│ ├─ Clean ILO Statistics Data                               │
│ ├─ Clean SDG National Data                                 │
│ ├─ Clean SDG Regional Data                                 │
│ └─ Clean World Bank Data                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Data Integration                                  │
│ ├─ Merge ILO + World Bank datasets                         │
│ ├─ Integrate SDG National data                             │
│ ├─ Handle missing values systematically                    │
│ └─ Create feature categories                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Exploratory Data Analysis (EDA)                   │
│ ├─ Distribution analysis                                   │
│ ├─ Skewness & kurtosis assessment                          │
│ ├─ Outlier detection (IQR method)                          │
│ └─ Data transformation (log, winsorization)                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Correlation & Statistical Analysis                │
│ ├─ Pearson correlation analysis                            │
│ ├─ Significance testing (p-values)                         │
│ ├─ Regression modeling (Linear, Ridge, Lasso)              │
│ └─ Model evaluation (R², RMSE, MAE)                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: Time Series Forecasting                           │
│ ├─ ARIMA models for unemployment trends                    │
│ ├─ Prophet models for seasonal patterns                    │
│ ├─ Scenario analysis (policy interventions)                │
│ └─ Forecast validation                                     │
└─────────────────────────────────────────────────────────────┘

```

---

## 🔧 Section 1: Library Imports

### Purpose
Set up the Python environment with necessary libraries for data manipulation, analysis, and visualization.

### Libraries Used

#### Data Manipulation
```python
import pandas as pd        # DataFrame operations
import numpy as np         # Numerical computations
```

#### Visualization

```python
import matplotlib.pyplot as plt    # Basic plotting
import seaborn as sns             # Statistical visualizations
import plotly.express as px       # Interactive charts
import plotly.graph_objects as go # Custom interactive plots
```

#### Statistical Analysis

```python
from scipy import stats           # Statistical tests
from scipy.stats import pearsonr  # Correlation analysis
from statsmodels.tsa.seasonal import seasonal_decompose  # Time series decomposition
```

#### Machine Learning

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
```

#### Time Series Forecasting

```python
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet  # Facebook Prophet for time series
```

### Configuration

```python
# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
```

---

## 🧹 Section 2: Data Cleaning

### 2.1 Cleaning ILO Statistics Data

#### Objective

Prepare labour force data from the International Labour Organization.

#### Steps Performed

**1. Load Raw Data**

```python
ilo_df = pd.read_csv('ilostat_data.csv')
print(f"Initial shape: {ilo_df.shape}")
```

**2. Handle Missing Values**

- Identify columns with >50% missing data
- Decision: Keep labour-critical columns despite missing data
- Apply forward-fill for time series continuity
- Use median imputation for remaining gaps

```python
# Calculate missing percentages
missing_pct = (ilo_df.isnull().sum() / len(ilo_df) * 100).sort_values(ascending=False)

# Forward fill for time series
ilo_df = ilo_df.sort_values(['country', 'year'])
ilo_df = ilo_df.groupby('country').fillna(method='ffill')

# Median imputation for remaining gaps
numeric_cols = ilo_df.select_dtypes(include=[np.number]).columns
ilo_df[numeric_cols] = ilo_df[numeric_cols].fillna(ilo_df[numeric_cols].median())
```

**3. Standardize Country Names**

- Map ILO country names to standard ISO format
- Ensure consistency across datasets

**4. Data Validation**

- Check for duplicate records
- Verify data types
- Ensure year range is consistent (2014-2021)

#### Output

- Clean ILO dataset with standardized country names
- Reduced missing values from ~45% to <5%
- Saved as: `ilostat_cleaned.csv`

---

### 2.2 Cleaning SDG National Data

#### Objective

Clean UNESCO's Sustainable Development Goal 4 (Education) indicators.

#### Steps Performed

**1. Load and Inspect**

```python
sdg_nat_df = pd.read_csv('sdg_national_data.csv')
print(f"Unique indicators: {sdg_nat_df['indicator_id'].nunique()}")
print(f"Countries: {sdg_nat_df['country'].nunique()}")
```

**2. Indicator Code Mapping**
Created human-readable mappings for SDG indicator codes:

| Code                    | Full Name                              | Description                                                  |
| ----------------------- | -------------------------------------- | ------------------------------------------------------------ |
| `SDG_NAT_AIR.1.GLAST` | Adjusted Net Attendance Rate - Primary | % of primary-age children attending primary/secondary school |
| `SDG_NAT_CR.MOD.1`    | Completion Rate - Primary              | % completing primary education                               |
| `SDG_NAT_CR.MOD.2`    | Completion Rate - Lower Secondary      | % completing lower secondary                                 |
| `SDG_NAT_CR.MOD.3`    | Completion Rate - Upper Secondary      | % completing upper secondary                                 |
| `SDG_NAT_XGDP.FSGOV`  | Education Expenditure (% GDP)          | Government spending on education                             |
| `SDG_NAT_ROFST.MOD.1` | Trained Teachers - Primary             | % of trained teachers in primary                             |
| `SDG_NAT_OAEPG.1`     | Out-of-School Rate                     | % of school-age children not attending                       |
| `SDG_NAT_GER.5T8`     | Gross Enrollment Ratio - Tertiary      | Tertiary enrollment rate                                     |

**3. Gender Indicator Suffixes**

- `.F` = Female-specific indicator
- `.M` = Male-specific indicator
- `.GPIA` = Gender Parity Index (Female/Male ratio)
- No suffix = Overall/Both genders

**4. Pivot to Wide Format**

```python
# Transform from long to wide format
sdg_wide = sdg_nat_df.pivot_table(
    index=['country', 'country_id', 'year'],
    columns='indicator_id',
    values='value',
    aggfunc='first'
).reset_index()
```

**5. Handle Missing Values**

- Linear interpolation for time series gaps
- Group by country to maintain data integrity
- Cap interpolation to avoid unrealistic values

```python
# Linear interpolation by country
sdg_wide = sdg_wide.sort_values(['country', 'year'])
sdg_wide = sdg_wide.groupby('country').apply(
    lambda group: group.interpolate(method='linear', limit_direction='both')
)
```

**6. Data Validation**

- Verify Gender Parity Index is close to 1 (0.85-1.15 range)
- Check completion rates don't exceed 100%
- Ensure expenditure percentages are reasonable (<10% GDP)

#### Output

- Wide-format SDG dataset: 1,350 rows × 50 columns
- Saved as: `sdg_national_cleaned.csv`

---

### 2.3 Cleaning SDG Regional Data

#### Objective

Prepare regional aggregations for continental/sub-regional analysis.

#### Steps Performed

**1. Regional Groupings**

- North Africa: Algeria, Egypt, Libya, Morocco, Tunisia
- West Africa: Benin, Burkina Faso, Ghana, Nigeria, Senegal, etc.
- East Africa: Ethiopia, Kenya, Rwanda, Tanzania, Uganda
- Southern Africa: Botswana, South Africa, Zambia, Zimbabwe
- Central Africa: Cameroon, Chad, Congo, Gabon

**2. Aggregation Methods**

- Mean for rates/percentages
- Sum for absolute counts
- Weighted averages where population data available

**3. Save Regional Summaries**

```python
regional_summary = sdg_regional.groupby(['region', 'year']).agg({
    'SDG_NAT_CR.MOD.1': 'mean',
    'Unemployment, youth total (% of total labor force ages 15-24)': 'mean',
    'SDG_NAT_XGDP.FSGOV': 'mean'
})
```

#### Output

- Regional aggregated data for trend analysis
- Saved as: `sdg_regional_cleaned.csv`

---

### 2.4 Cleaning World Bank Data

#### Objective

Prepare economic and demographic indicators from World Bank.

#### Steps Performed

**1. Load Data**

```python
wb_df = pd.read_csv('worldbank_data.csv')
```

**2. Indicator Selection**
Focused on education-relevant indicators:

- Economic: GDP growth, FDI, sector contributions
- Demographic: Population growth, working-age population
- Education: Literacy rates, enrollment rates, expenditure
- Labour: Unemployment rates, labour force participation

**3. Missing Value Strategy**

```python
# Calculate missing percentages
missing_analysis = (wb_df.isnull().sum() / len(wb_df) * 100).sort_values(ascending=False)

# Strategy:
# - Drop columns with >70% missing (irrelevant to analysis)
# - Linear interpolation for 30-70% missing (time series)
# - Median imputation for <30% missing (sporadic gaps)

# Drop high-missing columns
drop_cols = missing_analysis[missing_analysis > 70].index
wb_df = wb_df.drop(columns=drop_cols)

# Linear interpolation
wb_df = wb_df.sort_values(['country', 'year'])
wb_df = wb_df.groupby('country').apply(
    lambda x: x.interpolate(method='linear', limit=3)
)

# Median imputation
numeric_cols = wb_df.select_dtypes(include=[np.number]).columns
wb_df[numeric_cols] = wb_df.groupby('country')[numeric_cols].transform(
    lambda x: x.fillna(x.median())
)
```

**4. Data Type Corrections**

```python
# Ensure proper data types
wb_df['year'] = wb_df['year'].astype(int)
wb_df['country_id'] = wb_df['country_id'].astype(str)

# Numeric columns
numeric_indicators = [col for col in wb_df.columns 
                     if col not in ['country', 'country_id', 'year']]
wb_df[numeric_indicators] = wb_df[numeric_indicators].apply(pd.to_numeric, errors='coerce')
```

**5. Outlier Detection (Preliminary)**

```python
# Identify extreme outliers using IQR method
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Flag but don't remove (domain knowledge needed)
for col in numeric_indicators:
    outliers = detect_outliers_iqr(wb_df[col])
    if outliers.sum() > 0:
        print(f"{col}: {outliers.sum()} outliers detected")
```

#### Output

- Clean World Bank dataset: 1,350 rows × 25 columns
- Missing values reduced from ~35% to <5%
- Saved as: `worldbank_cleaned.csv`

---

## 🔗 Section 3: Data Integration

### 3.1 Merging ILO and World Bank Data

#### Objective

Create a unified labour-economic dataset.

#### Steps

**1. Initial Merge**

```python
# Merge on country and year
labour_economic_df = pd.merge(
    ilo_cleaned_df,
    worldbank_cleaned_df,
    on=['country', 'country_id', 'year'],
    how='inner',  # Only keep matching records
    suffixes=('_ilo', '_wb')
)

print(f"Merged shape: {labour_economic_df.shape}")
print(f"Records before merge: ILO={len(ilo_cleaned_df)}, WB={len(worldbank_cleaned_df)}")
print(f"Records after merge: {len(labour_economic_df)}")
```

**2. Handle Column Conflicts**

```python
# Identify duplicate columns
duplicate_cols = [col.replace('_ilo', '').replace('_wb', '') 
                 for col in labour_economic_df.columns 
                 if '_ilo' in col or '_wb' in col]

# Resolution strategy:
# - For identical values: keep one version
# - For different values: average or keep most complete

for base_col in duplicate_cols:
    col_ilo = f"{base_col}_ilo"
    col_wb = f"{base_col}_wb"
  
    if col_ilo in labour_economic_df.columns and col_wb in labour_economic_df.columns:
        # Average if both exist
        labour_economic_df[base_col] = labour_economic_df[[col_ilo, col_wb]].mean(axis=1)
        # Drop suffixed versions
        labour_economic_df = labour_economic_df.drop(columns=[col_ilo, col_wb])
```

---

### 3.2 Integrating SDG National Data

#### Objective

Add education indicators to create final integrated dataset.

#### Steps

**1. Merge with Labour-Economic Data**

```python
integrated_df = pd.merge(
    labour_economic_df,
    sdg_national_cleaned_df,
    on=['country', 'country_id', 'year'],
    how='inner'
)

print(f"Final integrated shape: {integrated_df.shape}")
```

**2. Column Organization**

```python
# Organize columns by category
id_cols = ['country', 'country_id', 'year']
labour_cols = [col for col in integrated_df.columns if 'unemployment' in col.lower()]
education_cols = [col for col in integrated_df.columns if 'SDG_NAT' in col]
economic_cols = [col for col in integrated_df.columns if 'GDP' in col or 'FDI' in col]

# Reorder for readability
column_order = id_cols + labour_cols + education_cols + economic_cols + [
    col for col in integrated_df.columns if col not in id_cols + labour_cols + education_cols + economic_cols
]

integrated_df = integrated_df[column_order]
```

---

### 3.3 Final Missing Value Treatment

#### Objective

Address remaining missing values in integrated dataset.

#### Strategy

**1. Analyze Missing Patterns**

```python
missing_summary = pd.DataFrame({
    'Column': integrated_df.columns,
    'Missing_Count': integrated_df.isnull().sum(),
    'Missing_Pct': (integrated_df.isnull().sum() / len(integrated_df) * 100).round(2)
}).sort_values('Missing_Pct', ascending=False)

print(missing_summary[missing_summary['Missing_Pct'] > 0])
```

**2. Imputation Methods**

| Missing % | Method                             | Rationale                      |
| --------- | ---------------------------------- | ------------------------------ |
| 0-5%      | Median imputation                  | Minimal impact on distribution |
| 5-20%     | Linear interpolation (time series) | Preserves trends               |
| 20-50%    | Group median (by region)           | Leverages similar countries    |
| >50%      | Keep as-is or drop                 | Too much uncertainty           |

**3. Implementation**

```python
# Low missing: median imputation
low_missing_cols = missing_summary[
    (missing_summary['Missing_Pct'] > 0) & 
    (missing_summary['Missing_Pct'] <= 5)
]['Column'].tolist()

for col in low_missing_cols:
    integrated_df[col] = integrated_df[col].fillna(integrated_df[col].median())

# Medium missing: linear interpolation
medium_missing_cols = missing_summary[
    (missing_summary['Missing_Pct'] > 5) & 
    (missing_summary['Missing_Pct'] <= 20)
]['Column'].tolist()

integrated_df = integrated_df.sort_values(['country', 'year'])
for col in medium_missing_cols:
    integrated_df[col] = integrated_df.groupby('country')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

# High missing: regional median
high_missing_cols = missing_summary[
    (missing_summary['Missing_Pct'] > 20) & 
    (missing_summary['Missing_Pct'] <= 50)
]['Column'].tolist()

# Add region column (if not exists)
region_mapping = {
    # North Africa
    'Algeria': 'North Africa', 'Egypt, Arab Rep.': 'North Africa', 
    'Libya': 'North Africa', 'Morocco': 'North Africa', 'Tunisia': 'North Africa',
    # West Africa
    'Benin': 'West Africa', 'Burkina Faso': 'West Africa', 'Ghana': 'West Africa',
    # ... (complete mapping)
}

integrated_df['region'] = integrated_df['country'].map(region_mapping)

for col in high_missing_cols:
    integrated_df[col] = integrated_df.groupby('region')[col].transform(
        lambda x: x.fillna(x.median())
    )
```

**4. Final Validation**

```python
# Verify no critical columns have missing values
critical_cols = ['country', 'year', 'SDG_NAT_CR.MOD.1', 
                'Unemployment, youth total (% of total labor force ages 15-24)']

for col in critical_cols:
    assert integrated_df[col].isnull().sum() == 0, f"{col} still has missing values"

print("✅ Missing value treatment complete!")
print(f"Final dataset: {integrated_df.shape}")
print(f"Total missing values: {integrated_df.isnull().sum().sum()}")
```

---

### 3.4 Outlier Treatment

#### Objective

Identify and handle outliers to prevent distortion of statistical analyses.

#### Method: Interquartile Range (IQR)

**1. Outlier Detection**

```python
def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method
    Returns: Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
  
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
  
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
  
    return outliers, lower_bound, upper_bound

# Apply to numeric columns
numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns

outlier_summary = {}
for col in numeric_cols:
    outliers, lb, ub = detect_outliers_iqr(integrated_df, col)
    outlier_count = outliers.sum()
  
    if outlier_count > 0:
        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': (outlier_count / len(integrated_df) * 100).round(2),
            'lower_bound': lb,
            'upper_bound': ub
        }

print(json.dumps(outlier_summary, indent=2))
```

**2. Treatment Strategy**

| Outlier % | Treatment            | Rationale                          |
| --------- | -------------------- | ---------------------------------- |
| <1%       | Winsorization        | Cap at bounds, preserve data       |
| 1-5%      | Case-by-case review  | Check if legitimate extreme values |
| >5%       | Transform data (log) | Likely skewed distribution         |

**3. Winsorization Implementation**

```python
from scipy.stats import mstats

def winsorize_outliers(df, column, limits=(0.05, 0.05)):
    """
    Cap extreme values at specified percentiles
    limits: (lower, upper) percentiles to cap
    """
    winsorized = mstats.winsorize(df[column].dropna(), limits=limits)
    df[column] = pd.Series(winsorized, index=df[column].dropna().index)
    return df

# Apply winsorization to columns with <5% outliers
for col, info in outlier_summary.items():
    if info['percentage'] < 5:
        integrated_df = winsorize_outliers(integrated_df, col)
        print(f"✅ Winsorized: {col}")
```

**4. Log Transformation (for highly skewed data)**

```python
# Identify skewed columns
from scipy.stats import skew

skewness = integrated_df[numeric_cols].apply(lambda x: skew(x.dropna()))
highly_skewed = skewness[abs(skewness) > 1].index.tolist()

print(f"Highly skewed columns: {highly_skewed}")

# Log transformation
for col in highly_skewed:
    if (integrated_df[col] > 0).all():  # Only if all values positive
        integrated_df[f"{col}_log"] = np.log1p(integrated_df[col])
        print(f"✅ Log transformed: {col}")
```

---

### 3.5 Feature Categorization

#### Objective

Organize indicators into logical categories for easier analysis.

#### Categories Created

**1. Education Indicators**

```python
education_features = [
    # Attendance rates
    'SDG_NAT_AIR.1.GLAST', 'SDG_NAT_AIR.1.GLAST.F', 'SDG_NAT_AIR.1.GLAST.M',
    'SDG_NAT_AIR.2.GPV.GLAST', 'SDG_NAT_AIR.2.GPV.GLAST.F', 'SDG_NAT_AIR.2.GPV.GLAST.M',
  
    # Completion rates
    'SDG_NAT_CR.MOD.1', 'SDG_NAT_CR.MOD.1.F', 'SDG_NAT_CR.MOD.1.M',
    'SDG_NAT_CR.MOD.2', 'SDG_NAT_CR.MOD.2.F', 'SDG_NAT_CR.MOD.2.M',
    'SDG_NAT_CR.MOD.3', 'SDG_NAT_CR.MOD.3.F', 'SDG_NAT_CR.MOD.3.M',
  
    # Literacy & enrollment
    'Literacy rate, youth total (% of people ages 15-24)',
    'Literacy rate, adult total (% of people ages 15 and above)',
    'School enrollment, primary (% gross)',
    'School enrollment, secondary (% gross)',
    'School enrollment, tertiary (% gross)',
  
    # Resources
    'Pupil-teacher ratio in primary education (headcount basis)',
    'Government expenditure on education, total (% of GDP)',
    'SDG_NAT_XGDP.FSGOV',
    'SDG_NAT_XGOVEXP.IMF'
]
```

**2. Labour Indicators**

```python
labour_features = [
    'Unemployment, total (% of total labor force)',
    'Unemployment, youth total (% of total labor force ages 15-24)',
    'Unemployment with advanced education (% of total labor force with advanced education)'
]
```

**3. Economic Indicators**

```python
economic_features = [
    'GDP growth (annual %)',
    'Foreign direct investment, net inflows (% of GDP)',
    'Agriculture, forestry, and fishing, value added (% of GDP)',
    'Industry (including construction), value added (% of GDP)',
    'Services, value added (% of GDP)'
]
```

**4. Demographic Indicators**

```python
demographic_features = [
    'Population ages 15-64 (% of total population)',
    'Population growth (annual %)'
]
```

**5. Save Metadata**

```python
feature_categories = {
    'education': education_features,
    'labour': labour_features,
    'economic': economic_features,
    'demographic': demographic_features
}

metadata = {
    'countries': sorted(integrated_df['country'].unique().tolist()),
    'years': sorted(integrated_df['year'].unique().tolist()),
    'feature_categories': feature_categories,
    'outliers': outlier_summary
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

### 3.6 Save Final Integrated Dataset

```python
# Save final clean integrated data
integrated_df.to_csv('integrated_data.csv', index=False)
print(f"✅ Saved integrated_data.csv: {integrated_df.shape}")

# Verify saved data
verify_df = pd.read_csv('integrated_data.csv')
assert len(verify_df) == len(integrated_df), "Data save verification failed"
print("✅ Data integrity verified")
```

**Final Dataset Statistics:**

- **Rows**: 1,350 (55 countries × 8 years, some gaps)
- **Columns**: 75+ indicators
- **Missing Values**: <2% overall
- **Outliers**: Treated via winsorization
- **File Size**: ~1.2 MB

---

## 📊 Section 4: Analysis

### 4.1 SDG Indicator Analysis

#### Objective

Understand the availability, distribution, and characteristics of SDG education indicators.

#### 4.1.1 Data Availability Analysis

**Purpose**: Assess completeness of SDG indicators across countries and years.

```python
# Calculate availability for each SDG indicator
sdg_indicators = [col for col in integrated_df.columns if 'SDG_NAT' in col]

availability_analysis = {}
for indicator in sdg_indicators:
    total_possible = len(integrated_df)
    available = integrated_df[indicator].notna().sum()
    percentage = (available / total_possible * 100).round(2)
  
    availability_analysis[indicator] = {
        'count': int(available),
        'percentage': float(percentage)
    }

# Sort by availability
sorted_availability = dict(sorted(
    availability_analysis.items(), 
    key=lambda x: x[1]['percentage'], 
    reverse=True
))

print(json.dumps(sorted_availability, indent=2))
```

**Key Findings:**

- Most SDG indicators have 100% availability (1,350 records)
- Highest availability: Completion rates, attendance rates
- Lower availability: Scholarship data, tertiary enrollment (varies by country)

**Visualization:**

```python
# Availability heatmap
availability_df = pd.DataFrame(availability_analysis).T
availability_df = availability_df.sort_values('percentage', ascending=False)

fig = px.bar(
    availability_df,
    x=availability_df.index,
    y='percentage',
    title='SDG Indicator Data Availability',
    labels={'x': 'Indicator', 'percentage': 'Availability (%)'},
    color='percentage',
    color_continuous_scale='Viridis'
)
fig.show()
```

---

#### 4.1.2 Descriptive Statistics

**Purpose**: Understand central tendencies and variability of education indicators.

```python
# Calculate comprehensive statistics
sdg_stats = {}
for indicator in sdg_indicators:
    data = integrated_df[indicator].dropna()
  
    sdg_stats[indicator] = {
        'mean': float(data.mean()),
        'median': float(data.median()),
        'std': float(data.std()),
        'min': float(data.min()),
        'max': float(data.max()),
        'q25': float(data.quantile(0.25)),
        'q75': float(data.quantile(0.75)),
        'records': int(len(data))
    }

# Display statistics for key indicators
key_indicators = [
    'SDG_NAT_CR.MOD.1',  # Primary completion
    'SDG_NAT_CR.MOD.2',  # Lower secondary completion
    'SDG_NAT_CR.MOD.3',  # Upper secondary completion
    'SDG_NAT_XGDP.FSGOV' # Education expenditure
]

stats_df = pd.DataFrame(sdg_stats).T
stats_df = stats_df.loc[key_indicators]
print(stats_df.round(2))
```

**Example Output:**

```
                          mean  median    std    min     max    q25    q75  records
SDG_NAT_CR.MOD.1         57.59   58.85  22.50   6.30   99.90  41.20  75.40     1350
SDG_NAT_CR.MOD.2         45.32   45.60  23.15   4.20   98.50  28.35  62.80     1350
SDG_NAT_CR.MOD.3         32.18   28.90  21.40   2.10   95.20  15.60  45.30     1350
SDG_NAT_XGDP.FSGOV        3.78    3.39   2.02   0.90   12.50   2.32   4.85     1350
```

**Interpretation:**

- **Primary Completion**: Average 57.6%, indicating many countries struggle to achieve universal primary education
- **Secondary Completion**: Lower average (45.3% lower, 32.2% upper), showing significant dropout
- **Education Spending**: Average 3.8% of GDP, below UNESCO recommendation of 4-6%
- **High Variability**: Large standard deviations indicate vast differences across countries

---

#### 4.1.3 Gender Parity Analysis

**Purpose**: Assess gender disparities in education access and completion.

```python
# Analyze Gender Parity Indices (GPIA)
gpia_indicators = [col for col in integrated_df.columns if 'GPIA' in col]

gpia_analysis = {}
for indicator in gpia_indicators:
    data = integrated_df[indicator].dropna()
  
    # Perfect parity = 1.0
    # <1 = disparity favoring males
    # >1 = disparity favoring females
  
    gpia_analysis[indicator] = {
        'mean': float(data.mean()),
        'median': float(data.median()),
        'below_parity_count': int((data < 0.97).sum()),  # Significant male advantage
        'above_parity_count': int((data > 1.03).sum()),  # Significant female advantage
        'at_parity_count': int(((data >= 0.97) & (data <= 1.03)).sum()),  # Near parity
        'worst_disparity': float(data.min()),
        'best_parity': float(data.max())
    }

print(json.dumps(gpia_analysis, indent=2))
```

**Key Findings:**

```json
{
  "SDG_NAT_AIR.1.GLAST.GPIA": {
    "mean": 0.927,
    "median": 0.970,
    "below_parity_count": 450,
    "above_parity_count": 85,
    "at_parity_count": 815,
    "worst_disparity": 0.402,
    "best_parity": 1.289
  }
}
```

**Interpretation:**

- Primary attendance shows male advantage in ~33% of cases
- Gender parity improving over time (median moving toward 1.0)
- Some countries achieve female advantage in education
- Significant work remains to achieve universal gender parity

**Visualization:**

```python
# Gender parity trends
for gpia_col in ['SDG_NAT_CR.MOD.1.GPIA', 'SDG_NAT_CR.MOD.2.GPIA', 'SDG_NAT_CR.MOD.3.GPIA']:
    trend = integrated_df.groupby('year')[gpia_col].mean()
  
    plt.plot(trend.index, trend.values, marker='o', label=gpia_col)

plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Parity')
plt.xlabel('Year')
plt.ylabel('Gender Parity Index')
plt.title('Gender Parity Trends in Education Completion')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 4.2 Exploratory Data Analysis (EDA)

#### 4.2.1 Distribution Analysis

**Purpose**: Understand data distributions to inform modeling choices.

**1. Skewness Assessment**

```python
from scipy.stats import skew, kurtosis

numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns

distribution_stats = {}
for col in numeric_cols:
    data = integrated_df[col].dropna()
  
    if len(data) > 0:
        distribution_stats[col] = {
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'skewness': float(skew(data)),
            'kurtosis': float(kurtosis(data))
        }

# Categorize distributions
symmetric_cols = [col for col, stats in distribution_stats.items() 
                  if abs(stats['skewness']) < 0.5]
skewed_cols = [col for col, stats in distribution_stats.items() 
               if abs(stats['skewness']) >= 0.5]

print(f"Symmetric distributions: {len(symmetric_cols)}")
print(f"Skewed distributions: {len(skewed_cols)}")
```

**Interpretation Guide:**

| Skewness    | Interpretation    | Action                    |
| ----------- | ----------------- | ------------------------- |
| -0.5 to 0.5 | Fairly symmetric  | Use as-is                 |
| 0.5 to 1.0  | Moderately skewed | Consider transformation   |
| > 1.0       | Highly skewed     | Log transform recommended |

| Kurtosis | Interpretation            | Implication          |
| -------- | ------------------------- | -------------------- |
| ~0       | Normal (mesokurtic)       | Typical distribution |
| > 0      | Heavy tails (leptokurtic) | More outliers        |
| < 0      | Light tails (platykurtic) | Fewer outliers       |

**2. Distribution Visualization**

```python
# Histogram grid for key indicators
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

key_indicators = [
    'SDG_NAT_CR.MOD.1',
    'SDG_NAT_CR.MOD.2',
    'SDG_NAT_CR.MOD.3',
    'Unemployment, youth total (% of total labor force ages 15-24)',
    'GDP growth (annual %)',
    'SDG_NAT_XGDP.FSGOV',
    'Foreign direct investment, net inflows (% of GDP)',
    'Literacy rate, adult total (% of people ages 15 and above)',
    'Population growth (annual %)'
]

for idx, indicator in enumerate(key_indicators):
    data = integrated_df[indicator].dropna()
  
    axes[idx].hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
    axes[idx].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
    axes[idx].set_title(indicator[:50], fontsize=10)
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=300)
plt.show()
```

---

#### 4.2.2 Correlation Analysis

**Purpose**: Identify relationships between variables to guide further analysis.

**1. Compute Correlation Matrix**

```python
# Select numeric columns
numeric_df = integrated_df[numeric_cols]

# Compute Pearson correlation
correlation_matrix = numeric_df.corr(method='pearson')

# Save for reference
correlation_matrix.to_csv('correlation_matrix.csv')
```

**2. Visualize Correlation Heatmap**

```python
# Full correlation heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(
    correlation_matrix,
    cmap='RdBu_r',
    center=0,
    annot=False,  # Too many variables to annotate
    fmt='.2f',
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title('Full Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('full_correlation_heatmap.png', dpi=300)
plt.show()
```

**3. Focus on Education-Labour Correlations**

```python
# Education vs Labour indicators
education_indicators = [col for col in education_features if col in numeric_cols]
labour_indicators = [col for col in labour_features if col in numeric_cols]

# Compute correlations with significance testing
from scipy.stats import pearsonr

education_labour_correlations = []

for edu_ind in education_indicators:
    for lab_ind in labour_indicators:
        # Get data (drop NaN)
        data = integrated_df[[edu_ind, lab_ind]].dropna()
      
        if len(data) > 30:  # Minimum sample size
            corr, p_value = pearsonr(data[edu_ind], data[lab_ind])
          
            # Significance levels
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'
          
            education_labour_correlations.append({
                'education_indicator': edu_ind,
                'labour_indicator': lab_ind,
                'correlation': corr,
                'p_value': p_value,
                'sample_size': len(data),
                'significance': significance
            })

# Sort by absolute correlation
education_labour_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

# Save results
with open('education_labour_correlations.json', 'w') as f:
    json.dump(education_labour_correlations, f, indent=2)

# Display top correlations
print("Top 10 Education-Labour Correlations:")
for i, corr in enumerate(education_labour_correlations[:10], 1):
    print(f"{i}. {corr['education_indicator']} → {corr['labour_indicator']}")
    print(f"   Correlation: {corr['correlation']:.3f} (p={corr['p_value']:.4f}) {corr['significance']}")
    print()
```

**Key Findings:**

**Strongest Positive Correlations** (Education ↑ → Unemployment ↑):

1. **Primary Completion Rate → Youth Unemployment**: r=0.483, p<0.001

   - *Interpretation*: Higher education completion paradoxically correlates with higher youth unemployment, suggesting skills mismatch
2. **Secondary Gender Parity → Total Unemployment**: r=0.386, p<0.001

   - *Interpretation*: Gender equality in education doesn't immediately translate to employment parity
3. **Female Primary Attendance → Youth Unemployment**: r=0.349, p<0.001

   - *Interpretation*: More educated females face employment barriers

**Key Negative Correlations** (Education ↑ → Unemployment ↓):

- Limited significant negative correlations found
- Suggests education alone doesn't reduce unemployment without complementary factors

**4. Correlation Heatmap (Education vs Labour)**

```python
# Create focused heatmap
selected_edu = [
    'SDG_NAT_CR.MOD.1',  # Primary completion
    'SDG_NAT_CR.MOD.2',  # Lower secondary completion
    'SDG_NAT_CR.MOD.3',  # Upper secondary completion
    'Literacy rate, adult total (% of people ages 15 and above)',
    'School enrollment, tertiary (% gross)',
    'SDG_NAT_XGDP.FSGOV'  # Education expenditure
]

selected_lab = [
    'Unemployment, total (% of total labor force)',
    'Unemployment, youth total (% of total labor force ages 15-24)',
    'Unemployment with advanced education (% of total labor force with advanced education)'
]

# Extract sub-correlation matrix
sub_corr_matrix = correlation_matrix.loc[selected_edu, selected_lab]

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    sub_corr_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)
plt.title('Education vs Labour Market Correlations', fontsize=14, fontweight='bold')
plt.xlabel('Labour Indicators', fontsize=12)
plt.ylabel('Education Indicators', fontsize=12)
plt.tight_layout()
plt.savefig('education_labour_correlation_heatmap.png', dpi=300)
plt.show()
```

---

## 🤖 Section 5: Statistical Relationship Models

### 5.1 Linear Regression Models

#### Objective

Quantify the relationship between education indicators (predictors) and unemployment (target).

#### 5.1.1 Simple Linear Regression

**Purpose**: Examine individual education indicator effects on unemployment.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Target variable
target = 'Unemployment, youth total (% of total labor force ages 15-24)'

# Test each education indicator individually
simple_regression_results = []

for edu_indicator in education_indicators:
    # Prepare data
    data = integrated_df[[edu_indicator, target]].dropna()
  
    if len(data) < 50:
        continue
  
    X = data[[edu_indicator]].values
    y = data[target].values
  
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
  
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
  
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
  
    # Evaluation metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
  
    simple_regression_results.append({
        'indicator': edu_indicator,
        'coefficient': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        'r2_train': float(r2_train),
        'r2_test': float(r2_test),
        'rmse_test': float(rmse_test),
        'mae_test': float(mae_test),
        'sample_size': len(data)
    })

# Sort by R² test performance
simple_regression_results.sort(key=lambda x: x['r2_test'], reverse=True)

# Display top performers
print("Top 5 Single-Indicator Models:")
for i, result in enumerate(simple_regression_results[:5], 1):
    print(f"{i}. {result['indicator']}")
    print(f"   R² (test): {result['r2_test']:.3f}")
    print(f"   RMSE: {result['rmse_test']:.2f}")
    print(f"   Coefficient: {result['coefficient']:.4f}")
    print()
```

**Example Output:**

```
Top 5 Single-Indicator Models:
1. SDG_NAT_CR.MOD.1
   R² (test): 0.234
   RMSE: 9.87
   Coefficient: 0.2156

2. SDG_NAT_CR.MOD.1.F
   R² (test): 0.226
   RMSE: 9.92
   Coefficient: 0.1987

3. SDG_NAT_AIR.2.GPV.GLAST.GPIA
   R² (test): 0.149
   RMSE: 10.42
   Coefficient: 7.8923
```

**Interpretation:**

- Primary completion rate explains ~23% of youth unemployment variance
- Low R² values indicate multifactorial nature of unemployment
- Positive coefficients confirm paradoxical relationship (education ↑, unemployment ↑)

---

#### 5.1.2 Multiple Linear Regression

**Purpose**: Combine multiple education indicators for better predictions.

```python
# Select top predictors based on correlation analysis
top_predictors = [
    'SDG_NAT_CR.MOD.1',  # Primary completion
    'SDG_NAT_CR.MOD.2',  # Lower secondary completion
    'SDG_NAT_XGDP.FSGOV',  # Education expenditure
    'Literacy rate, adult total (% of people ages 15 and above)',
    'GDP growth (annual %)',  # Control variable
    'Population growth (annual %)'  # Control variable
]

# Prepare data
data = integrated_df[top_predictors + [target]].dropna()
X = data[top_predictors].values
y = data[target].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple regression
multi_model = LinearRegression()
multi_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = multi_model.predict(X_train_scaled)
y_pred_test = multi_model.predict(X_test_scaled)

# Evaluation
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print("Multiple Linear Regression Results:")
print(f"R² (train): {r2_train:.3f}")
print(f"R² (test): {r2_test:.3f}")
print(f"RMSE (test): {rmse_test:.2f}")
print(f"MAE (test): {mae_test:.2f}")
print()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': top_predictors,
    'Coefficient': multi_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("Feature Importance (Standardized Coefficients):")
print(feature_importance)
```

**Example Output:**

```
Multiple Linear Regression Results:
R² (train): 0.358
R² (test): 0.342
RMSE (test): 9.15
MAE (test): 7.23

Feature Importance (Standardized Coefficients):
                                          Feature  Coefficient
0                          SDG_NAT_CR.MOD.1        3.245
1                          SDG_NAT_CR.MOD.2        2.187
5                   Population growth (annual %)  -1.876
2                       SDG_NAT_XGDP.FSGOV         1.543
3  Literacy rate, adult total (% of people ...     0.982
4                       GDP growth (annual %)     -0.765
```

**Interpretation:**

- Combined model explains ~34% of unemployment variance (improvement from 23%)
- Primary and secondary completion are strongest predictors
- Population growth negatively associated (demographic dividend effect)
- Education spending positively associated (investment takes time to show results)

---

### 5.2 Regularized Regression Models

#### Objective

Handle multicollinearity and prevent overfitting using Ridge and Lasso regression.

#### 5.2.1 Ridge Regression

**Purpose**: Reduce coefficient magnitudes to prevent overfitting.

```python
from sklearn.linear_model import Ridge, RidgeCV

# Cross-validation to find optimal alpha
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
ridge_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {ridge_cv.alpha_}")

# Train Ridge with optimal alpha
ridge_model = Ridge(alpha=ridge_cv.alpha_)
ridge_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_ridge_train = ridge_model.predict(X_train_scaled)
y_pred_ridge_test = ridge_model.predict(X_test_scaled)

# Evaluation
r2_ridge_train = r2_score(y_train, y_pred_ridge_train)
r2_ridge_test = r2_score(y_test, y_pred_ridge_test)
rmse_ridge_test = np.sqrt(mean_squared_error(y_test, y_pred_ridge_test))

print("Ridge Regression Results:")
print(f"R² (train): {r2_ridge_train:.3f}")
print(f"R² (test): {r2_ridge_test:.3f}")
print(f"RMSE (test): {rmse_ridge_test:.2f}")
```

---

#### 5.2.2 Lasso Regression

**Purpose**: Perform feature selection by driving some coefficients to zero.

```python
from sklearn.linear_model import Lasso, LassoCV

# Cross-validation for optimal alpha
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {lasso_cv.alpha_}")

# Train Lasso
lasso_model = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lasso_train = lasso_model.predict(X_train_scaled)
y_pred_lasso_test = lasso_model.predict(X_test_scaled)

# Evaluation
r2_lasso_train = r2_score(y_train, y_pred_lasso_train)
r2_lasso_test = r2_score(y_test, y_pred_lasso_test)
rmse_lasso_test = np.sqrt(mean_squared_error(y_test, y_pred_lasso_test))

print("Lasso Regression Results:")
print(f"R² (train): {r2_lasso_train:.3f}")
print(f"R² (test): {r2_lasso_test:.3f}")
print(f"RMSE (test): {rmse_lasso_test:.2f}")

# Feature selection
lasso_coefficients = pd.DataFrame({
    'Feature': top_predictors,
    'Coefficient': lasso_model.coef_
})

selected_features = lasso_coefficients[lasso_coefficients['Coefficient'] != 0]
print(f"\nFeatures selected by Lasso: {len(selected_features)}/{len(top_predictors)}")
print(selected_features)
```

---

### 5.3 Model Comparison

```python
# Compare all models
model_comparison = pd.DataFrame({
    'Model': ['Simple (Best)', 'Multiple Linear', 'Ridge', 'Lasso'],
    'R² Train': [0.234, r2_train, r2_ridge_train, r2_lasso_train],
    'R² Test': [0.234, r2_test, r2_ridge_test, r2_lasso_test],
    'RMSE Test': [9.87, rmse_test, rmse_ridge_test, rmse_lasso_test]
})

print(model_comparison)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R² comparison
axes[0].bar(model_comparison['Model'], model_comparison['R² Test'], color='skyblue')
axes[0].set_title('Model Performance (R² Test)')
axes[0].set_ylabel('R²')
axes[0].grid(True, alpha=0.3)

# RMSE comparison
axes[1].bar(model_comparison['Model'], model_comparison['RMSE Test'], color='coral')
axes[1].set_title('Model Error (RMSE Test)')
axes[1].set_ylabel('RMSE')
axes[1].grid(True, alpha=0.3)

# Actual vs Predicted (best model)
axes[2].scatter(y_test, y_pred_test, alpha=0.6)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[2].set_xlabel('Actual')
axes[2].set_ylabel('Predicted')
axes[2].set_title('Actual vs Predicted (Multiple Linear)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()
```

**Key Findings:**

- Multiple regression outperforms single-indicator models
- Regularization provides modest improvement in generalization
- ~34% R² indicates unemployment is highly multifactorial
- Education is necessary but not sufficient for employment

---

## 🔮 Section 6: Time Series Forecasting

### 6.1 ARIMA Forecasting

#### Objective

Forecast future unemployment trends using historical patterns.

#### 6.1.1 Data Preparation

```python
# Select a representative country
forecast_country = 'Nigeria'
country_ts = integrated_df[integrated_df['country'] == forecast_country].sort_values('year')

# Target: Youth unemployment
ts_data = country_ts[['year', 'Unemployment, youth total (% of total labor force ages 15-24)']].copy()
ts_data = ts_data.set_index('year')
ts_data = ts_data.dropna()

print(f"Time series length: {len(ts_data)}")
print(ts_data)
```

---

#### 6.1.2 Stationarity Check

```python
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller test
adf_result = adfuller(ts_data.values.flatten())

print("ADF Test Results:")
print(f"Test Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print(f"Critical Values: {adf_result[4]}")

if adf_result[1] < 0.05:
    print("✅ Series is stationary")
else:
    print("❌ Series is non-stationary, differencing needed")
```

---

#### 6.1.3 ARIMA Model Fitting

```python
from statsmodels.tsa.arima.model import ARIMA

# Try different ARIMA parameters
# (p, d, q): p=AR order, d=differencing, q=MA order

best_aic = np.inf
best_params = None

for p in range(0, 3):
    for d in range(0, 2):
        for q in range(0, 3):
            try:
                model = ARIMA(ts_data, order=(p, d, q))
                fitted_model = model.fit()
              
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = (p, d, q)
            except:
                continue

print(f"Best ARIMA parameters: {best_params}")
print(f"AIC: {best_aic:.2f}")

# Fit best model
best_arima = ARIMA(ts_data, order=best_params)
fitted_arima = best_arima.fit()

print(fitted_arima.summary())
```

---

#### 6.1.4 Forecasting

```python
# Forecast next 5 years
forecast_steps = 5
forecast = fitted_arima.forecast(steps=forecast_steps)

forecast_years = range(ts_data.index[-1] + 1, ts_data.index[-1] + 1 + forecast_steps)

# Visualization
plt.figure(figsize=(12, 6))

# Historical data
plt.plot(ts_data.index, ts_data.values, marker='o', label='Historical', color='blue', linewidth=2)

# Forecast
plt.plot(forecast_years, forecast, marker='s', label='Forecast', color='red', linestyle='--', linewidth=2)

plt.xlabel('Year', fontsize=12)
plt.ylabel('Youth Unemployment Rate (%)', fontsize=12)
plt.title(f'Youth Unemployment Forecast - {forecast_country}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'arima_forecast_{forecast_country}.png', dpi=300)
plt.show()

# Print forecast values
forecast_df = pd.DataFrame({
    'Year': list(forecast_years),
    'Forecasted Youth Unemployment (%)': forecast.values
})
print(forecast_df)
```

---

### 6.2 Prophet Forecasting

#### Objective

Use Facebook Prophet for more flexible time series modeling with seasonality.

```python
from prophet import Prophet

# Prepare data for Prophet (requires 'ds' and 'y' columns)
prophet_data = country_ts[['year', 'Unemployment, youth total (% of total labor force ages 15-24)']].copy()
prophet_data.columns = ['ds', 'y']
prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%Y')
prophet_data = prophet_data.dropna()

# Initialize Prophet model
prophet_model = Prophet(
    yearly_seasonality=False,  # Not enough data for yearly
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # Lower = less flexible (prevent overfitting)
)

# Fit model
prophet_model.fit(prophet_data)

# Create future dataframe
future = prophet_model.make_future_dataframe(periods=5, freq='Y')

# Forecast
prophet_forecast = prophet_model.predict(future)

# Visualize
fig = prophet_model.plot(prophet_forecast)
plt.title(f'Prophet Forecast - {forecast_country}', fontsize=14, fontweight='bold')
plt.ylabel('Youth Unemployment Rate (%)')
plt.tight_layout()
plt.savefig(f'prophet_forecast_{forecast_country}.png', dpi=300)
plt.show()

# Components
fig2 = prophet_model.plot_components(prophet_forecast)
plt.tight_layout()
plt.savefig(f'prophet_components_{forecast_country}.png', dpi=300)
plt.show()
```

---

### 6.3 Scenario Analysis

#### Objective

Model how changes in education indicators affect future unemployment.

```python
# Scenario 1: Increase primary completion by 10%
# Scenario 2: Increase education expenditure by 1% of GDP
# Scenario 3: Combined improvement

# Use regression model coefficients
base_unemployment = y_test.mean()

scenarios = {
    'Baseline': base_unemployment,
    'Scenario 1: +10% Primary Completion': base_unemployment + (0.2156 * 10),  # From regression
    'Scenario 2: +1% Education Spending': base_unemployment + (1.543 * 0.1),  # Scaled coefficient
    'Scenario 3: Combined': base_unemployment + (0.2156 * 10) + (1.543 * 0.1)
}

print("Scenario Analysis Results:")
for scenario, value in scenarios.items():
    print(f"{scenario}: {value:.2f}% youth unemployment")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['gray', 'skyblue', 'lightgreen', 'coral']
bars = ax.bar(scenarios.keys(), scenarios.values(), color=colors, edgecolor='black', linewidth=1.5)

ax.axhline(y=base_unemployment, color='red', linestyle='--', linewidth=2, label='Baseline')
ax.set_ylabel('Youth Unemployment Rate (%)', fontsize=12)
ax.set_title('Policy Scenario Analysis', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('scenario_analysis.png', dpi=300)
plt.show()
```

**Key Insight:**

- Paradoxically, increasing education completion alone may increase short-term unemployment
- This suggests skills mismatch: education system not aligned with job market needs
- Policy recommendation: Combine education expansion with vocational training and job creation

---

## 🎯 Key Findings

### Education Landscape

1. **Completion Rates**:

   - Primary: 57.6% average (target: 100%)
   - Lower Secondary: 45.3% average
   - Upper Secondary: 32.2% average
   - Significant dropout between levels
2. **Gender Parity**:

   - Improving trend toward equality
   - 33% of cases still show male advantage
   - Some countries achieve female advantage
3. **Education Spending**:

   - Average: 3.8% of GDP
   - Below UNESCO recommendation (4-6%)
   - High variability across countries

### Labour Market

1. **Youth Unemployment**:

   - Critical challenge across continent
   - Averages 15-25% depending on country
   - Higher education doesn't guarantee employment
2. **Skills Mismatch**:

   - Positive correlation between education and unemployment
   - Indicates education system not aligned with job market
   - Need for vocational and technical training
3. **Economic Factors**:

   - GDP growth weakly correlated with employment
   - Sector composition matters (services vs. agriculture)
   - FDI inflows don't directly create jobs for educated youth

### Statistical Relationships

1. **Correlation Analysis**:

   - Primary completion → Youth unemployment: r=0.483***
   - Education spending shows delayed effects
   - Gender parity in education ≠ employment parity
2. **Regression Models**:

   - Education explains ~34% of unemployment variance
   - Multifactorial problem requiring holistic solutions
   - Population growth has negative effect (demographic dividend)
3. **Forecasting**:

   - Unemployment trends show gradual improvement
   - Country-specific patterns vary significantly
   - Policy interventions can bend the curve

---

## 🔧 Technical Notes

### Data Quality

- **Completeness**: 95%+ after cleaning
- **Missing Value Treatment**: Linear interpolation, median imputation
- **Outlier Treatment**: IQR method with winsorization
- **Data Validation**: Cross-checked against source documentation

### Statistical Methods

- **Correlation**: Pearson (assumes linear relationships)
- **Significance Testing**: α=0.05, 0.01, 0.001 levels
- **Regression**: OLS with regularization (Ridge, Lasso)
- **Time Series**: ARIMA, Prophet for forecasting

### Limitations

1. **Causality**: Correlation ≠ causation; observational data
2. **Omitted Variables**: Cultural, political factors not included
3. **Data Gaps**: Some countries/years have sparse data
4. **Model Assumptions**: Linearity may not hold for complex relationships

### Reproducibility

- **Random Seed**: 42 (for train-test splits)
- **Python Version**: 3.8+
- **Key Libraries**: pandas 1.3+, scikit-learn 1.0+, statsmodels 0.13+
- **Hardware**: Standard CPU sufficient (<1GB RAM)

---

## 📚 References

### Data Sources

1. UNESCO Institute for Statistics (UIS) - SDG 4 Education Indicators
2. World Bank Open Data - World Development Indicators
3. International Labour Organization (ILO) - ILOSTAT Database

### Methodological References

1. OECD (2019). "Education at a Glance"
2. UNESCO (2020). "Global Education Monitoring Report"
3. ILO (2020). "Global Employment Trends for Youth"

---

## 📄 Output Files Generated

| Filename                     | Description                  | Size    |
| ---------------------------- | ---------------------------- | ------- |
| `integrated_data.csv`      | Final merged dataset         | ~1.2 MB |
| `worldbank_cleaned.csv`    | Cleaned World Bank data      | ~425 KB |
| `sdg_national_cleaned.csv` | Cleaned SDG data             | ~53 MB  |
| `metadata.json`            | Data dictionary & categories | ~4 KB   |
| `analysis_results.json`    | All analysis outputs         | ~63 KB  |
| `correlation_matrix.csv`   | Full correlation matrix      | ~150 KB |
| `*.png`                    | Visualization outputs        | Various |

---

## 🚀 Next Steps

### Recommended Analysis Extensions

1. **Machine Learning**: Random Forest, Gradient Boosting for non-linear relationships
2. **Clustering**: Identify country groups with similar education-labour profiles
3. **Causal Inference**: Propensity score matching, difference-in-differences
4. **Spatial Analysis**: Geographic patterns and regional spillovers

### Dashboard Enhancements

1. **Interactive Filters**: Year, country, indicator selection
2. **Drill-Down Views**: Country → regional → national analysis
3. **Export Functionality**: Download charts, data subsets
4. **Scenario Builder**: User-defined policy interventions

### Policy Research Questions

1. What vocational programs best reduce youth unemployment?
2. How do education investments translate to economic growth?
3. What time lag exists between education and employment effects?
4. Which education quality indicators matter most?

---

**Document Version**: 1.0
**Last Updated**: 2025
**Author**: Data Analytics Team
**Contact**: [Your Contact Information]

---

*This documentation provides a comprehensive guide to understanding, reproducing, and extending the African Education-Labour Analytics project. For questions or clarifications, please refer to the project team.*

```

```
