# Authors
\<Adiba Akter\>
\<Richard\>
\<John\>
\<Tyson\>

# Data Domain

The data we are using is on exoplanets from the NASA Exoplanet Archive. It includes various attributes of known exoplanets, such as their names, host stars, orbital characteristics, and physical parameters like mass and radius. The table is designed for researchers and enthusiasts to analyze and compare the properties of these distant worlds.

The first 88 rows of the csv file contain full descriptions of the columns shorthand names. So to make the file readable for panda, those rows have to be removed.

# Importing Libraries

The following cell imports all libraries that will be used for this notebook.


```python
import numpy as np
import scipy as sp
import pandas as pd
# IMPORTANT: Allows this notebook to get data if on Colab.
# Comment out if running locally.
#from google.colab import drive
```

# Data Preprocessing

# Gathering the Data

The following code cell will import the data from the CSV file and put it in a Pandas `DataFrame`.

Here, `pd.read_csv` function is used from the Pandos library which will read the CSV file the data is currently in. The parameter `comment` tells the function that there are comments in the CSV file and will handle them appropriately to prevent errors and garbage from being put into the `DataFrame` `df_data`.


```python
# Path to the file. If on Colab, use the appropriate Colab package to import
#  the data.
path = "/PSCompPars_2024.09.17_17.11.11.csv"

# IMPORTANT: If running on Google Colab, uncomment this line and use instead.
#  You will need to enable permissions on your account to use.
# If you are doing this locally, please use the file path above instead.
#drive.mount('/content/drive')
#path = "/content/drive/My Drive/School/SacStateClasses/CSC177/CSC177_GroupFolder/1_DataPreprocessing/PSCompPars_2024.09.17_17.11.11.csv"

path = "./ORI_PSCompPars_2024.09.17_17.11.11.csv" #from Adiba, helpful when you download in zip from google drive, if working on collab, comment this line and use other path above

df_data = pd.read_csv(path, comment='#', na_values=['', np.nan])


print("Display information about the DataFrame:")
print(df_data.dtypes)
print("\nShow contents of the data:")
df_data
```

# Preprocessing the Data

## Remove Duplicate Data

The first operation will be to remove any possible duplicate data from the `DataFrame`.


```python
# Code for duplicate data.
dups = df_data.duplicated()
print(f"Number of duplicate rows = {dups.sum()}")
dups
# Uncomment this line if the data has any duplicats
#df_data = df_data.drop_duplicates()
```

The following cell will remove any duplicates using the `duplicated` method of the `DataFrame`.


```python
df_data.drop_duplicates(inplace=True)
dups = df_data.duplicated()
print(f"Duplicates: {dups.sum()}")
dups
```

## Fill In Missing Values

The next operation is to fill in any possible missing data values. Several columns where the uncertainty is null, we replace them with the mean uncertainty. There are a few steps that must be taken however:

1. Replace any missing data points with `np.nan`.
2. Replace `np.nan` with the *mean* of that attribute.


```python
# This will go through the entire DataFrame and replace any empty attributes
#  with NaN using np.nan
df_data = df_data.replace('', np.nan)
df_data

```

After replacing missing attribute values with NaN, next is to replace those attribute values the mean for each feature. This is done by using the `mean` and `fillna` methods of the `df_data` object.


```python
sr_mean_vals = df_data.mean(numeric_only=True) # Series that contains the mean
df_data.fillna(sr_mean_vals, inplace=True)
print(f"df_data shape: {df_data.shape}")
df_data
```

## Task 3: Data Transformation
In this task, we will be transforming the data by applying the following techniques:

1. **Encoding categorical variables**: Categorical features need to be transformed into numerical values. We use one-hot encoding for this task to handle categories in a way that machine learning algorithms can process.

2. **Normalization**: Numerical data is normalized using `StandardScaler()` to ensure that all features are on a similar scale, which is particularly important for algorithms sensitive to feature scaling.

3. **Handling Outliers**: We identify and remove outliers using the Interquartile Range (IQR) method. This helps to ensure that extreme values do not unduly influence the model’s performance.



## Step 1: Encoding Categorical Variables


```python
# Task 3: Encoding Categorical Variables
# The categorical columns are encoded using LabelEncoder

from sklearn.preprocessing import LabelEncoder

# List of categorical columns to encode
categorical_columns = ['discoverymethod', 'disc_facility', 'pl_name', 'hostname']

# Encoding each categorical column using LabelEncoder
le = LabelEncoder()
for col in categorical_columns:
    df_data[col] = le.fit_transform(df_data[col].astype(str))

# Display the first few rows of the encoded data
print("Encoded categorical data:")
df_data[categorical_columns].head()
```

## Step 2: Normalizing Numerical Features


```python
# Task 3: Normalizing Numerical Features
# Using StandardScaler to normalize the numerical columns

from sklearn.preprocessing import StandardScaler

# Selecting numerical columns for scaling
numerical_columns = df_data.select_dtypes(include=['float64', 'int64']).columns

# Initializing the StandardScaler
scaler = StandardScaler()

# Fitting the scaler to the numerical data and transforming it
df_data[numerical_columns] = scaler.fit_transform(df_data[numerical_columns])

# Display the first few rows of the normalized data
print("Normalized data:")
df_data[numerical_columns].head()

```

## Step 3: Handling Outliers with the IQR Method


```python
# Task 3: Handling Outliers using the Interquartile Range (IQR)
# We detect outliers and filter them out

# Calculating Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_data[numerical_columns].quantile(0.25)
Q3 = df_data[numerical_columns].quantile(0.75)

# Calculating the Interquartile Range (IQR)
IQR = Q3 - Q1

# Defining lower and upper bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Removing rows where any numerical column has outliers
df_data_cleaned = df_data[~((df_data[numerical_columns] < lower_bound) | 
                            (df_data[numerical_columns] > upper_bound)).any(axis=1)]

# Display the shape and a few rows of the cleaned dataset
print(f"Data after handling outliers, original shape: {df_data.shape}, cleaned shape: {df_data_cleaned.shape}")
df_data_cleaned.head()

```
