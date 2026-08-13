# Algerian Forest Fires Dataset: Comprehensive Data Cleaning

This notebook serves as a detailed guide to cleaning the Algerian Forest Fires dataset, a dataset notably challenging due to its 'dirty' nature, often requiring extensive preprocessing before analysis. It's an excellent resource for understanding common data cleaning issues and their resolutions.

## Data Loading and Initial Inspection

The dataset was loaded and inspected to understand its initial structure and identify immediate issues such as incorrect headers and mixed data types.

```python
import pandas as pd
import numpy as np

# Load the dataset, skipping the initial header row as it contains metadata
dataset = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', header=1)

# Display the first few rows to observe the data
dataset.head()

# Get a concise summary of the DataFrame, including data types and non-null values
dataset.info()
```

## Handling Missing Values and Structural Issues

The dataset presented several challenges, including embedded metadata rows and inconsistent data types, leading to many missing values.

### Identifying and Addressing Metadata Rows

A specific row (`Sidi-Bel Abbes Region Dataset`) was found to contain dataset metadata rather than actual data, causing `NaN` values across columns. This row needed to be identified and removed.

```python
# Identify rows with any missing values
dataset[dataset.isnull().any(axis=1)]

# Introduce a 'Region' column to distinguish between the two regional datasets
dataset.loc[:122, "Region"] = 0 # Bejaia Region
dataset.loc[122:, "Region"] = 1 # Sidi-Bel Abbes Region
df = dataset # Create a working copy

# Convert the 'Region' column to integer type
df[['Region']] = df[['Region']].astype(int)

# Re-check for null values after adding 'Region' to confirm it's not contributing to NaNs
df.isnull().sum()

# Drop rows containing any missing values. This effectively removes the metadata row.
df = df.dropna().reset_index(drop=True)

# The row at index 122 (which was originally the 'Sidi-Bel Abbes Region Dataset' header) was removed.
# The new row at index 122 after dropna is an actual data row. 
# However, an additional header row was still present at the original index 122, which becomes current index 122 after initial dropna.
# It also needs to be removed.
df.iloc[[122]]

# Remove the problematic header row that shifted to index 122 after the previous dropna
df = df.drop(122).reset_index(drop=True)

# Verify removal
df.iloc[[122]]
```

### Cleaning Column Names

Many column names contained leading or trailing whitespace, which can cause issues during analysis. These were stripped for consistency.

```python
# Display current column names
df.columns

# Strip whitespace from all column names
df.columns = df.columns.str.strip()

# Verify cleaned column names
df.columns
```

## Data Type Conversion

Several columns, although containing numerical data, were incorrectly identified as `object` (string) types. These needed to be converted to appropriate numeric types (`int` or `float`).

```python
# Convert 'day', 'month', 'year', 'Temperature', 'RH', 'Ws' to integer type
df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)

# Get columns that are still of 'object' type (likely numerical but imported as strings)
objects = [features for features in df.columns if df[features].dtypes == 'O']

# Iterate through object columns and convert them to float, excluding 'Classes'
for i in objects:
  if i != 'Classes': # 'Classes' is intentionally kept as object/string for now
    df[i] = df[i].astype(float)

# Final check of data types after conversions
df.info()
```

### Cleaning and Encoding the Target Variable ('Classes')

The 'Classes' column, representing fire occurrences, had inconsistent string values (e.g., 'fire ', 'fire', 'not fire '). These were standardized and then encoded into a binary numerical format.

```python
# Create a copy for further exploratory data analysis (EDA) without modifying the original cleaned df
df_copy = df.drop(['day', 'month', 'year'], axis=1) # Drop date components as they might not be needed for direct modeling

# Check the unique values in the 'Classes' column before encoding
df_copy['Classes'].value_counts()

# Encode 'Classes' column: 'not fire' becomes 0, 'fire' becomes 1. 
# The .str.contains() method handles variations like 'fire ' or 'not fire '.
df_copy['Classes'] = np.where(df_copy['Classes'].str.contains('not fire'), 0, 1)

# Verify the new value counts for 'Classes'
df_copy['Classes'].value_counts()

# Display the head of the DataFrame copy to see the cleaned 'Classes' column
df_copy.head()
```

## Saving the Cleaned Dataset

Finally, the thoroughly cleaned dataset was saved to a new CSV file for future use, ensuring all preprocessing steps are captured.

```python
# Save the cleaned DataFrame to a new CSV file
df.to_csv('Algerian_forest_fires_cleaned_dataset.csv', index=False)
```

This comprehensive cleaning process addresses common issues like missing values, incorrect data types, inconsistent formatting, and structural anomalies, preparing the dataset for robust analysis and model training.
