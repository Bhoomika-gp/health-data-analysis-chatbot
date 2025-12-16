# GenAI Health Data Analysis Pipeline
This project implements a complete data analysis workflow. It goes from raw Excel file ingestion to an interactive, AI-powered web application. The pipeline allows natural language queries of structured health data. It uses the Google Gemini LLM.
## Project Architecture & Flow
The system processes data in stages. It ends with a Streamlit web application:
### Data Ingestion & Preprocessing: 
Raw Excel datasets (.xlsm) are loaded and cleaned using pandas. Key preprocessing includes handling missing values, removing duplicates, and creating new features (e.g., BMI_Category, Age_Group).
### Database Storage: 
Preprocessed dataframes are saved into normalized SQLite databases. This allows for efficient retrieval.
### GenAI Pipeline: 
The core HealthDataGenAIPipeline executes user queries:
An LLM generates schema-aware SQL queries from natural language prompts.
pandasql executes the generated SQL against the SQLite databases.
The LLM interprets the query results and creates a concise, human-readable report.
### User Interface: 
A Streamlit application provides an interactive front-end. Users can input queries, view generated SQL, see raw results, and analyze evaluation metrics.



