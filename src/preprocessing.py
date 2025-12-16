# ============================================================================ #
# HEALTH DATA PREPROCESSING PIPELINE
# ============================================================================ #

# src/preprocessing.py
import os
import pandas as pd
import numpy as np
import warnings
import sqlite3

warnings.filterwarnings('ignore')
#STEP 1
# ================= PATH CONFIGURATION ================= #
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # parent of src/
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

DATASET1_PATH = os.path.join(DATA_DIR, 'dataset1.xlsm')
DATASET2_PATH = os.path.join(DATA_DIR, 'dataset2.xlsm')

DB_PATH_1 = os.path.join(PROJECT_ROOT, 'preprocessing_dataset_1.db')
DB_PATH_2 = os.path.join(PROJECT_ROOT, 'preprocessing_dataset_2.db')

# ================= LOAD DATASETS ================= #
try:
    df1 = pd.read_excel(DATASET1_PATH)
    print(f"✓ Dataset 1 loaded: {df1.shape}")
    df2 = pd.read_excel(DATASET2_PATH)
    print(f"✓ Dataset 2 loaded: {df2.shape}")
except Exception as e:
    print(f"✗ Error loading datasets: {e}")
    exit()

# ============================================================================
# STEP 2: DATASET 1 PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("[STEP 2] Preprocessing Dataset 1 (Dynamic Missing Value Handling)")
print("="*70)

df1_clean = df1.copy()
print("✓ Created a copy of the original dataset for preprocessing")

# ---------------------- 2.1 Remove exact duplicates -----------------------
print("\n[2.1] Removing exact duplicate rows if any...")
dup_count = df1_clean.duplicated().sum()
print(f"✓ Found {dup_count} duplicate rows")
df1_clean.drop_duplicates(inplace=True)
print(f"✓ Dataset shape after removing duplicates: {df1_clean.shape}")

# ---------------------- 2.2 Unique Patient Constraint ---------------------
print("\n[2.2] Checking for duplicate Patient Numbers...")
if 'Patient_Number' in df1_clean.columns:
    dup_pat = df1_clean['Patient_Number'].duplicated().sum()
    if dup_pat > 0:
        df1_clean.drop_duplicates(subset=['Patient_Number'], inplace=True)
        print(f"✓ Removed {dup_pat} duplicate Patient Numbers")
    else:
        print("✓ No duplicate Patient Numbers found")

# ---------------------- 2.3 Check & handle missing values -----------------
print("\n[2.3] Checking columns with missing values...")
missing_summary = df1_clean.isnull().sum()
missing_percentage = (missing_summary / len(df1_clean)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing_summary, 'Percentage': missing_percentage})
missing_cols = missing_df[missing_df['Missing_Count'] > 0]

if missing_cols.empty:
    print("✓ No missing values found in any column")
else:
    print("✓ Columns with missing values:")
    print(missing_cols)

# Handle missing values dynamically
for col in missing_cols.index:
    if col == 'Pregnancy':
        df1_clean['Pregnancy'].fillna(df1_clean['Pregnancy'].mode()[0], inplace=True)
        df1_clean.loc[df1_clean['Sex'] == 0, 'Pregnancy'] = 0
        df1_clean['Pregnancy'] = df1_clean['Pregnancy'].astype(int)
        print(f"✓ Filled missing 'Pregnancy', fixed males, converted to int")
    elif col == 'Genetic_Pedigree_Coefficient':
        df1_clean['Genetic_Pedigree_Coefficient'].fillna(df1_clean['Genetic_Pedigree_Coefficient'].median(), inplace=True)
        print(f"✓ Filled missing 'Genetic_Pedigree_Coefficient' with median")
    elif col == 'alcohol_consumption_per_day':
        df1_clean['alcohol_consumption_per_day'].fillna(df1_clean['alcohol_consumption_per_day'].median(), inplace=True)
        df1_clean['alcohol_consumption_per_day'] = df1_clean['alcohol_consumption_per_day'].astype(int)
        print(f"✓ Filled missing 'alcohol_consumption_per_day' with median and converted to int")

# ---------------------- 2.4 Biological Validations ------------------------
print("\n[2.4] Performing biological validations...")
if 'Genetic_Pedigree_Coefficient' in df1_clean.columns:
    df1_clean['Genetic_Pedigree_Coefficient'] = df1_clean['Genetic_Pedigree_Coefficient'].clip(0, 1)
    print("✓ Clipped 'Genetic_Pedigree_Coefficient' to range [0–1]")

if 'Age' in df1_clean.columns:
    prev_rows = df1_clean.shape[0]
    df1_clean = df1_clean[(df1_clean['Age'] >= 0) & (df1_clean['Age'] <= 120)]
    removed_rows = prev_rows - df1_clean.shape[0]
    print(f"✓ Removed {removed_rows} unrealistic Age values")

invalid_preg = df1_clean[(df1_clean['Sex'] == 0) & (df1_clean['Pregnancy'] == 1)]
print(f"✓ Invalid pregnancy cases after fix: {len(invalid_preg)}")

# ---------------------- 2.5 Feature Engineering --------------------------
print("\n[2.5] Creating derived features...")
if 'BMI' in df1_clean.columns:
    df1_clean['BMI_Category'] = pd.cut(df1_clean['BMI'], bins=[0,18.5,25,30,60],
                                       labels=['Underweight','Normal','Overweight','Obese'])
    print("✓ Created 'BMI_Category' feature")

if 'Age' in df1_clean.columns:
    df1_clean['Age_Group'] = pd.cut(df1_clean['Age'], bins=[0,18,35,50,65,120],
                                    labels=['<18','18-35','36-50','51-65','>65'])
    print("✓ Created 'Age_Group' feature")

if 'Level_of_Stress' in df1_clean.columns:
    df1_clean['Stress_Label'] = df1_clean['Level_of_Stress'].map({1:'Low',2:'Normal',3:'High'})
    print("✓ Created 'Stress_Label' feature")

print("\n🎯 Dataset 1 Preprocessing Completed Successfully!")
print(f"Final Dataset Shape: {df1_clean.shape}")

# ============================================================================
# STEP 3: STORE DATASET 1 INTO SQLITE DATABASE
# ============================================================================
print("\n" + "="*70)
print("[STEP 3] Saving Dataset 1 to SQLite Database")
print("="*70)

db_path_1 = "preprocessing_dataset_1.db"
conn = sqlite3.connect(DB_PATH_1)
df1_clean.to_sql("preprocessing_dataset_1", conn, if_exists="replace", index=False)
conn.close()
print(f"📌 Dataset 1 successfully saved to SQLite DB: {db_path_1}")

# ============================================================================
# STEP 4: DATASET 2 PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("[STEP 4] Preprocessing Dataset 2 (Daily Activity Dataset)")
print("="*70)

df2_clean = df2.copy()
print("✓ Created a copy of the original dataset for preprocessing")

# ---------------------- 4.1 Remove duplicates --------------------------------
print("\n[4.1] Checking for duplicates...")
duplicates_2 = df2_clean.duplicated().sum()
print(f"✓ Found {duplicates_2} duplicate rows")
df2_clean.drop_duplicates(inplace=True)
print(f"✓ Dataset shape after removing duplicates: {df2_clean.shape}")

# ---------------------- 4.2 Unique patient-day combinations -----------------
print("\n[4.2] Checking for duplicate (Patient_Number, Day_Number) combinations...")
dup_combinations = df2_clean.duplicated(subset=['Patient_Number', 'Day_Number']).sum()
if dup_combinations > 0:
    df2_clean.drop_duplicates(subset=['Patient_Number', 'Day_Number'], inplace=True)
    print(f"✓ Removed {dup_combinations} duplicate combinations")
else:
    print("✓ No duplicate combinations found")

# ---------------------- 4.3 Missing value handling -------------------------
print("\n[4.3] Checking for missing values...")
missing_counts_2 = df2_clean.isnull().sum()
missing_percentage_2 = (missing_counts_2 / len(df2_clean)) * 100
missing_df_2 = pd.DataFrame({'Missing_Count': missing_counts_2, 'Percentage': missing_percentage_2})
missing_cols_2 = missing_df_2[missing_df_2['Missing_Count'] > 0]

if not missing_cols_2.empty:
    print(f"✓ Missing values found:\n{missing_cols_2}")

    # Fill Physical_activity dynamically
    if 'Physical_activity' in missing_cols_2.index:
        df2_clean['Physical_activity'] = df2_clean.groupby('Patient_Number')['Physical_activity'].fillna(method='ffill')
        df2_clean['Physical_activity'] = df2_clean.groupby('Patient_Number')['Physical_activity'].fillna(method='bfill')
        df2_clean['Physical_activity'].fillna(df2_clean['Physical_activity'].median(), inplace=True)
        print("✓ Filled missing 'Physical_activity'")
else:
    print("✓ No missing values found")

# ---------------------- 4.4 Validate Day_Number ---------------------------
print("\n[4.4] Validating Day_Number range...")
if 'Day_Number' in df2_clean.columns:
    invalid_days = df2_clean[(df2_clean['Day_Number'] < 1) | (df2_clean['Day_Number'] > 10)]
    if len(invalid_days) > 0:
        print(f"⚠ Found {len(invalid_days)} invalid Day_Number records")
    else:
        print("✓ All Day_Numbers are within range [1,10]")

# ---------------------- 4.5 Validate Physical_activity ---------------------
print("\n[4.5] Validating Physical_activity values...")
if 'Physical_activity' in df2_clean.columns:
    negative_activity = df2_clean[df2_clean['Physical_activity'] < 0]
    if len(negative_activity) > 0:
        df2_clean = df2_clean[df2_clean['Physical_activity'] >= 0]
        print(f"✓ Removed negative Physical_activity values")
    else:
        print("✓ No negative Physical_activity values found")

# ---------------------- 4.6 Feature Engineering --------------------------
print("\n[4.6] Feature Engineering...")
if 'Physical_activity' in df2_clean.columns:
    min_pa = df2_clean['Physical_activity'].min()
    max_pa = df2_clean['Physical_activity'].max()
    bins = [min_pa-1, 5000, 10000, 15000, max_pa+1]
    df2_clean['Activity_Level'] = pd.cut(df2_clean['Physical_activity'],
                                         bins=bins,
                                         labels=['Sedentary', 'Moderate', 'Active', 'Very Active'])
    print(f"✓ Created 'Activity_Level' categories")

# ---------------------- 4.7 Aggregate statistics per patient --------------
print("\n[4.7] Aggregating patient statistics...")
patient_stats = df2_clean.groupby('Patient_Number').agg({
    'Physical_activity': ['mean', 'median', 'std', 'min', 'max', 'sum']
}).reset_index()
patient_stats.columns = ['Patient_Number', 'Avg_Physical_Activity', 'Median_Physical_Activity',
                         'Std_Physical_Activity', 'Min_Physical_Activity', 'Max_Physical_Activity', 
                         'Total_Physical_Activity']
print(f"✓ Aggregated statistics created for {len(patient_stats)} patients")

print(f"\n🎯 Dataset 2 Preprocessing Completed Successfully!")
print(f"Final Dataset Shape: {df2_clean.shape}")
print(f"Patient Statistics Shape: {patient_stats.shape}")

# ============================================================================
# STEP 5: STORE DATASET 2 INTO SQLITE DATABASE
# ============================================================================
print("\n" + "="*70)
print("[STEP 5] Saving Dataset 2 to SQLite Database")
print("="*70)

db_path_2 = "preprocessing_dataset_2.db"
conn = sqlite3.connect(DB_PATH_2)
df2_clean.to_sql("activity_dataset_2", conn, if_exists="replace", index=False)
patient_stats.to_sql("activity_stats_2", conn, if_exists="replace", index=False)
conn.close()
print(f"📌 Dataset 2 saved to SQLite DB: {db_path_2}")

# ============================================================================
# STEP 6: FINAL DATA QUALITY REPORT
# ============================================================================
print("\n" + "="*70)
print("[STEP 6] Final Data Quality Report")
print("="*70)

# Dataset 1 Summary
print("\n--- Dataset 1 Summary ---")
print(f"Original shape: {df1.shape}")
print(f"Cleaned shape: {df1_clean.shape}")
print(f"Records removed: {df1.shape[0] - df1_clean.shape[0]}")
print(f"Missing values: {df1_clean.isnull().sum().sum()}")
print(f"Duplicate records: 0")
print(f"Columns: {list(df1_clean.columns)}")

# Dataset 2 Summary
print("\n--- Dataset 2 Summary ---")
print(f"Original shape: {df2.shape}")
print(f"Cleaned shape: {df2_clean.shape}")
print(f"Records removed: {df2.shape[0] - df2_clean.shape[0]}")
print(f"Missing values: {df2_clean.isnull().sum().sum()}")
print(f"Duplicate records: 0")
print(f"Unique patients: {df2_clean['Patient_Number'].nunique()}")
print(f"Columns: {list(df2_clean.columns)}")

# # ============================================================================
# # STEP 7: PREVIEW CLEANED DATA
# # ============================================================================
# print("\n" + "="*70)
# print("[STEP 7] Preview of Cleaned & Stored Data")
# print("="*70)

# print("\n--- Dataset 1 (Health Dataset 1) ---")
# print(df1_clean.head())

# print("\n--- Dataset 2 (Daily Activity Dataset) ---")
# print(df2_clean.head())

# print("\n--- Patient Activity Statistics ---")
# print(patient_stats.head())

# print("\n" + "="*70)
# print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
# print("="*70)
