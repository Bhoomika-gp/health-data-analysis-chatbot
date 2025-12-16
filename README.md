┌─────────────────────────┐
│  Excel Datasets         │
│  dataset1.xlsm          │
│  dataset2.xlsm          │
└──────────┬──────────────┘
           │
           │ Load (pandas.read_excel)
           ▼
┌─────────────────────────┐
│ Preprocessing Pipeline  │  ← src/preprocessing.py
│ Dataset 1 (df1)         │
│ - Remove duplicates     │
│ - Handle missing values │
│ - Enforce biological    │
│   constraints           │
│ - Feature engineering   │
│   (BMI_Category,        |
|     Age_Group,          |
|    Stress_Label)        │
│                         │
│ Dataset 2 (df2)         │
│ - Remove duplicates     │
│ - Handle missing PA     │
│ - Feature engineering   │
│   (Activity_Level)      │
│ - Aggregation/patient   │
└──────────┬──────────────┘
           │
           │ Save to SQLite
           ▼
┌─────────────────────────┐
│ SQLite Databases        │
│ preprocessing_dataset_1 │
│ activity_dataset_2      │
│ activity_stats_2        │
└──────────┬──────────────┘
           │
           │ Load (sqlite3 → pandas)
           ▼
┌─────────────────────────┐
│ GenAI Pipeline           │  ← src/genai_pipeline.py
│ HealthDataGenAIPipeline  │
│ - Load datasets from DB  │
│ - Initialize QueryExecutor 
       │
│ - Initialize GeminiHandler 
    (Google Gemini LLM)    │
│                           │
│ User Query: "What is avg BMI of male patients?" │
│                           │
│ 1️⃣ SQL Generation         │
│    - LLM generates SQL     │
│    - Schema-aware          │
│                           │
│ 2️⃣ Query Execution        │
│    - DuckDB executes SQL   │
│    - Returns dataframe     │
│                           │
│ 3️⃣ NL Response Generation │
│    - LLM receives results  │
│    - Creates concise report│
│                           │
│ 4️⃣ Evaluation              │
│    - Metrics: SQL success, │
│      data returned, column relevance, etc. │
└──────────┬──────────────┘
           │
           │ Output (dict)
           ▼
┌─────────────────────────┐
│ Streamlit Web Interface  │  ← app.py
│ - Query input / examples │
│ - Show SQL query         │
│ - Show raw data (CSV)    │
│ - Show AI analysis       │
│ - Evaluation metrics     │
│ - Quick stats tab        │
│ - Query history sidebar  │
└─────────────────────────┘
