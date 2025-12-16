"""
Optimized GenAI Health Data Analysis Pipeline
Pass only relevant columns to LLM without trimming rows to ensure accurate results.
"""

import os
import pandas as pd
import warnings
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import Dict, Any, List
import re
import sqlite3

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# ================= PATH CONFIGURATION ================= #
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # parent of src/
DB_PATH_1 = os.path.join(PROJECT_ROOT, 'preprocessing_dataset_1.db')
DB_PATH_2 = os.path.join(PROJECT_ROOT, 'preprocessing_dataset_2.db')

# ================= CONFIG ================= #
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = "gemini-2.5-flash"
    DB_PATH_1 = DB_PATH_1
    DB_PATH_2 = DB_PATH_2

# ============================================================================ #
# SCHEMA INFO
# ============================================================================ #
SCHEMA_INFO = """
DATABASE SCHEMA FOR HEALTH ANALYSIS:

TABLE 1: preprocessing_dataset_1 (Main Health Dataset - 2000 patients)
Columns:
- Patient_Number (int): Unique patient identifier
- Blood_Pressure_Abnormality (int): 0=Normal, 1=Abnormal
- Level_of_Hemoglobin (float): Hemoglobin level in g/dl
- Genetic_Pedigree_Coefficient (float): Disease genetic predisposition (0-1, closer to 1 = higher risk)
- Age (int): Patient age in years
- BMI (int): Body Mass Index
- Sex (int): 0=Male, 1=Female
- Pregnancy (int): 0=No, 1=Yes
- Smoking (int): 0=No, 1=Yes
- salt_content_in_the_diet (int): Salt intake in mg/day
- alcohol_consumption_per_day (int): Alcohol consumption in ml/day
- Level_of_Stress (int): 1=Low, 2=Normal, 3=High
- Chronic_kidney_disease (int): 0=No, 1=Yes
- Adrenal_and_thyroid_disorders (int): 0=No, 1=Yes
- BMI_Category (str): Underweight/Normal/Overweight/Obese
- Age_Group (str): <18/18-35/36-50/51-65/>65
- Stress_Label (str): Low/Normal/High

TABLE 2: activity_stats_2 (Aggregated Physical Activity - per patient)
Columns:
- Patient_Number (int): Unique patient identifier
- Avg_Physical_Activity (float): Average steps per day over 10 days
- Median_Physical_Activity (float): Median steps per day
- Std_Physical_Activity (float): Standard deviation of daily steps
- Min_Physical_Activity (float): Minimum steps in any day
- Max_Physical_Activity (float): Maximum steps in any day
- Total_Physical_Activity (float): Total steps over 10 days

TABLE 3: activity_dataset_2 (Daily Physical Activity - 10 days per patient)
Columns:
- Patient_Number (int): Unique patient identifier
- Day_Number (int): Day number (1-10)
- Physical_activity (float): Number of steps on that specific day
- Activity_Level (str): Sedentary/Moderate/Active/Very Active
"""

# ============================================================================ #
# QUERY EXECUTOR
# ============================================================================ #
class QueryExecutor:
    """Executes SQL-like queries on pandas DataFrames"""

    def __init__(self, df1: pd.DataFrame, df_stats: pd.DataFrame, df2: pd.DataFrame = None):
        """
        Args:.*
            df1: Health Dataset 1 (preprocessing_dataset_1)
            df_stats: Aggregated patient activity stats (activity_stats_2)
            df2: Daily activity dataset (activity_dataset_2)
        """
        self.df1 = df1
        self.df_stats = df_stats
        self.df2 = df2

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query using pandasql
        """
        try:
            import pandasql as psql

            # Map actual table names from SQLite DB to variable names
            local_vars = {
                'preprocessing_dataset_1': self.df1,
                'activity_stats_2': self.df_stats,
                'activity_dataset_2': self.df2 if self.df2 is not None else pd.DataFrame()
            }

            # Execute query
            result = psql.sqldf(query, local_vars)
            return result

        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query was: {query}")
            return pd.DataFrame()

    def execute_python_code(self, code: str) -> Any:
        """
        Execute Python code for data analysis in a safe environment
        """
        try:
            local_vars = {
                'df1': self.df1,
                'df_stats': self.df_stats,
                'df2': self.df2,
                'pd': pd,
                'np': np
            }

            exec(code, {"__builtins__": {}}, local_vars)

            return local_vars.get('result', "Code executed successfully")

        except Exception as e:
            print(f"Error executing Python code: {e}")
            return None


# ============================================================================ #
# GEMINI LLM HANDLER
# ============================================================================ #
class GeminiHandler:
    """Handles interactions with Google Gemini API"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_sql_query(self, user_question: str, schema: str) -> str:
        """
        Generate SQL query from natural language question
        """
        prompt = f"""You are an expert SQL query generator for health data analysis.

{schema}

USER QUESTION: {user_question}

INSTRUCTIONS:
1. Generate ONLY a valid SQL query that answers the user's question
2. Use proper JOIN syntax when combining tables
3. Use aggregate functions (AVG, COUNT, SUM) appropriately
4. Use WHERE clauses for filtering
5. Use GROUP BY when needed
6. Return ONLY the SQL query, no explanations or markdown
7. Do not include ```sql``` or any formatting

IMPORTANT:
- Table names: preprocessing_dataset_1, activity_stats_2, activity_dataset_2
- Always use proper column names from the schema
- For percentages, multiply by 100
- Use CASE statements for conditional logic

SQL QUERY:"""

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )

            sql_query = response.text.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            return sql_query

        except Exception as e:
            print(f"Error generating SQL query: {e}")
            return None

    def generate_response(self, user_question: str, query_results: Any) -> str:
        """
        Generate natural language response from query results
        """
        if isinstance(query_results, pd.DataFrame):
            if query_results.empty:
                results_str = "No data found matching the query criteria."
            else:
                results_str = query_results.to_string(index=False, max_rows=20)
        else:
            results_str = str(query_results)

        prompt = f"""You are a health data analyst providing insights to healthcare professionals.

USER QUESTION: {user_question}
QUERY RESULTS:
{results_str}

INSTRUCTIONS:
1. Provide a clear, professional natural language response
2. Include specific numbers and statistics from the results
3. Offer health insights and interpretations
4. If relevant, provide recommendations or observations
5. Use proper medical terminology
6. Structure your response with:
   - Direct answer to the question
   - Key findings with specific numbers
   - Health implications or insights
   - Recommendations (if applicable)
7. Be concise but comprehensive
8. Use emojis sparingly for visual organization (✅❌📊💡⚠️)

RESPONSE:"""

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]

            generate_content_config = types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.8,
            )

            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generate_content_config
            ):
                if chunk.text:
                    response_text += chunk.text

            return response_text.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating the response."


# ============================================================================ #
# RESPONSE EVALUATOR
# ============================================================================ #
class ResponseEvaluator:
    """Evaluates the quality of SQL query results and generated natural language responses."""

    @staticmethod
    def evaluate(query: str, results: pd.DataFrame, nl_response: str) -> dict:
        metrics = {}

        # 1️⃣ SQL execution success: 1 if results is not None
        metrics["SQL Success"] = 1 if results is not None else 0

        # 2️⃣ Data availability: 1 if the query returned a non-empty DataFrame
        metrics["Data Returned"] = 1 if isinstance(results, pd.DataFrame) and not results.empty else 0

        # 3️⃣ Check if the generated response contains any numeric insights
        metrics["Numerical Insight"] = 1 if any(ch.isdigit() for ch in nl_response) else 0

        # 4️⃣ Column relevance: fraction of result columns mentioned in the SQL query
        if results is not None and not results.empty:
            query_lower = query.lower()
            used_cols = [col.lower() for col in results.columns]
            match_count = sum(col in query_lower for col in used_cols)
            metrics["Column Relevance"] = round(match_count / len(results.columns), 2)
        else:
            metrics["Column Relevance"] = 0

        return metrics

# ============================================================================ #
# HEALTH DATA GENAI PIPELINE
# ============================================================================ #
class HealthDataGenAIPipeline:
    """
    Main pipeline for health data analysis using GenAI
    """

    def __init__(self, api_key: str):
        """Initialize the pipeline"""
        print("="*70)
        print("INITIALIZING HEALTH DATA GENAI PIPELINE")
        print("="*70)

        # 1️⃣ Initialize Gemini handler
        print("\n[1/3] Initializing Gemini AI...")
        self.gemini = GeminiHandler(api_key=api_key)
        print("✓ Gemini AI initialized")

        # 2️⃣ Load datasets from SQLite
        print("\n[2/3] Loading datasets from SQLite DBs...")
        import sqlite3

        # Load Dataset 1
        conn1 = sqlite3.connect(Config.DB_PATH_1)
        self.df1 = pd.read_sql("SELECT * FROM preprocessing_dataset_1", conn1)
        conn1.close()
        print(f"✓ Dataset 1 loaded: {self.df1.shape}")

        # Load Dataset 2 and aggregated stats
        conn2 = sqlite3.connect(Config.DB_PATH_2)
        self.df2 = pd.read_sql("SELECT * FROM activity_dataset_2", conn2)
        self.df_stats = pd.read_sql("SELECT * FROM activity_stats_2", conn2)
        conn2.close()
        print(f"✓ Dataset 2 loaded: {self.df2.shape}")
        print(f"✓ Activity stats loaded: {self.df_stats.shape}")

        # 3️⃣ Initialize query executor
        print("\n[3/3] Initializing query executor...")
        self.executor = QueryExecutor(self.df1, self.df_stats, self.df2)
        print("✓ Query executor initialized")

        print("\n" + "="*70)
        print("PIPELINE READY!")
        print("="*70 + "\n")

    def process_query(self, user_question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Process a user query end-to-end with evaluation
        """
        if verbose:
            print("\n" + "="*70)
            print(f"PROCESSING QUERY: {user_question}")
            print("="*70)

        # Step 1: Generate SQL query
        if verbose:
            print("\n[STEP 1] Generating SQL query...")
        sql_query = self.gemini.generate_sql_query(user_question, SCHEMA_INFO)

        if sql_query is None:
            return {
                "question": user_question,
                "sql_query": None,
                "results": None,
                "response": "I couldn't generate a query for your question.",
                "evaluation": {},
                "success": False
            }

        if verbose:
            print(f"✓ SQL Query Generated:\n{sql_query}")

        # Step 2: Execute SQL query
        if verbose:
            print("\n[STEP 2] Executing query...")
        results = self.executor.execute_query(sql_query)
        if isinstance(results, pd.DataFrame) and results.empty:
            if verbose:
                print("⚠ Query returned no results")
            results_summary = "No data found"
        else:
            results_summary = results
            if verbose:
                print(f"✓ Query executed successfully. Results shape: {results.shape}")

        # Step 3: Generate natural language response
        if verbose:
            print("\n[STEP 3] Generating natural language response...")
        nl_response = self.gemini.generate_response(user_question, results)
        if verbose:
            print("✓ Response generated")
            print("\n" + "="*70)
            print("FINAL RESPONSE:")
            print("="*70)
            print(nl_response)
            print("="*70 + "\n")

        # Step 4: Evaluate the response
        evaluation_scores = ResponseEvaluator.evaluate(sql_query, results, nl_response)

        return {
            "question": user_question,
            "sql_query": sql_query,
            "results": results,
            "response": nl_response,
            "evaluation": evaluation_scores,
            "success": True
        }

    def batch_process(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*70}")
            print(f"QUERY {i}/{len(questions)}")
            print(f"{'='*70}")
            result = self.process_query(question, verbose=True)
            results.append(result)
        return results
