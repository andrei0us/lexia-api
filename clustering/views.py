from django.http import JsonResponse
from django.db import connection
import pandas as pd
from sklearn.cluster import KMeans
from django.db import transaction

def perform_kmeans_and_update_cluster_data(request):
    try:
        with connection.cursor() as cursor:
            # --- TEMPORARY DEBUGGING CODE ---
            # This will show you the columns Django sees for students_progress_tbl
            try:
                # For MySQL
                cursor.execute("SHOW COLUMNS FROM students_progress_tbl;")
                # For PostgreSQL, use:
                # cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'students_progress_tbl' AND table_schema = DATABASE();") # DATABASE() for current DB
                columns_in_db = [col[0] for col in cursor.fetchall()]
                print(f"DEBUG: Columns found in students_progress_tbl (as seen by Django): {columns_in_db}")
                if 'retention' not in columns_in_db:
                    print("DEBUG: 'retention' column NOT found in Django's view of the table columns!")
            except Exception as debug_e:
                print(f"DEBUG: Could not execute SHOW COLUMNS query: {debug_e}")
            # --- END TEMPORARY DEBUGGING CODE ---

            # 1. Fetch aggregated performance metrics along with student_id
            cursor.execute("""
                SELECT
                    student_id,
                    AVG(accuracy) AS avg_accuracy,
                    AVG(consistency) AS avg_consistency,
                    AVG(speed) AS avg_speed,
                    AVG(retention) AS avg_retention,
                    AVG(problem_solving_skills) AS avg_problem_solving_skills,
                    AVG(vocabulary_range) AS avg_vocabulary_range
                FROM
                    students_progress_tbl
                GROUP BY
                    student_id
                HAVING
                    COUNT(accuracy) > 0 AND
                    COUNT(consistency) > 0 AND
                    COUNT(speed) > 0 AND
                    COUNT(retention) > 0 AND
                    COUNT(problem_solving_skills) > 0 AND
                    COUNT(vocabulary_range) > 0
            """)
            # ... (rest of your code remains the same)

    except Exception as e:
        print(f"Error during K-means execution: {e}")
        return JsonResponse({"error": f"An error occurred during K-means processing: {str(e)}"}, status=500)