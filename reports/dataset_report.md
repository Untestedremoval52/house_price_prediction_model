
# Dataset Audit Report

## Project

House Price Prediction using Linear Regression model

---

## Dataset

King County House Sales Dataset

---

## Dataset Overview

- Total Records: **21,613**
- Total Features: **22**
- Target Variable: **price**
- Dataset Size: **3.6 MB**

---

## Feature Composition

- Integer Features: **15**
- Float Features: **6**
- String Features: **1 (`date`)**

---

## Missing Values

| Feature   | Missing Values |
| --------- | -------------: |
| bedrooms  |             13 |
| bathrooms |             10 |
| Others    |              0 |

Only two features contain the missing values, accounting for a very small percentage of the dataset (nearly 0.07%).

---

## Duplicate Records

- Duplicate Rows: **0**

No duplicate observations were found upon dataset analysis.

---

## Statistical Highlights

- Minimum House Price: **75,000**
- Maximum House Price: **7,700,000**
- Maximum Living Area: **6,210 sq ft**
- Maximum Lot Size: **871,200 sq ft**

Large value ranges indicates the possible presence of outliers in the model which will be investigated during EDA.

---

## Initial Observations

- Dataset is clean and well-structured in its form.
- Only `date` column requires datatype conversion in the dataset.
- Missing values are very minimal in numbers.
- The dataset contains both structural and geographical features.
- Property IDs are not completely unique and they requires further investigation.
- Large value ranges suggest the possibility of potential outliers.

---

## Current Status

- Dataset isselected
- Folder structure is created
- Dataset successfully is loaded
- Initial dataset audit is completed

---

## Next Steps

- Investigate all the repeated property IDs
- Study the feature distributions of the dataset
- Detect all the outliers
- Design the preprocessing pipeline of the model

---
