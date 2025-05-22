from django.http import JsonResponse
from django.db import connection, transaction
import pandas as pd
from sklearn.cluster import KMeans
import traceback  # For detailed error logs

def perform_kmeans(request):
    """
    Performs K-means clustering on student performance data and updates the database.
    Intended to be triggered by a backend job or button in the dashboard.
    """
    try:
        with connection.cursor() as cursor:
            # --- DEBUG TEST QUERY ---
            try:
                cursor.execute("SELECT student_id, problem_solving_skills FROM students_progress_tbl LIMIT 1;")
                test_row = cursor.fetchone()
                print(f"DEBUG: Test fetch OK. Value: {test_row}")
            except Exception as debug_error:
                print("DEBUG: Problem accessing 'problem_solving_skills'")
                traceback.print_exc()
                return JsonResponse({"error": f"Problem accessing 'problem_solving_skills': {str(debug_error)}"}, status=500)
            # --- END DEBUG ---

            # 1. Fetch and aggregate student data
            cursor.execute("""
                SELECT
                    student_id,
                    AVG(accuracy) AS avg_accuracy,
                    AVG(consistency) AS avg_consistency,
                    AVG(speed) AS avg_speed,
                    AVG(retention) AS avg_retention,
                    AVG(problem_solving_skills) AS avg_problem_solving_skills,
                    AVG(vocabulary_range) AS avg_vocabulary_range
                FROM students_progress_tbl
                GROUP BY student_id
                HAVING
                    COUNT(accuracy) > 0 AND
                    COUNT(consistency) > 0 AND
                    COUNT(speed) > 0 AND
                    COUNT(retention) > 0 AND
                    COUNT(problem_solving_skills) > 0 AND
                    COUNT(vocabulary_range) > 0
            """)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=columns)

        # Separate features from student_id
        student_ids = df['student_id']
        features_df = df.drop(columns=['student_id'])

        # Drop NaNs
        features_df.dropna(inplace=True)
        if len(features_df) < 3:
            return JsonResponse({"message": "Not enough complete data for clustering."}, status=200)

        # Run KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(features_df)

        # Map clusters to labels
        label_map = {
            0: 'High Achiever',
            1: 'Developing Learner',
            2: 'Needs Support'
        }

        df['cluster_label'] = pd.Series(cluster_labels, index=features_df.index).map(label_map)

        # 2. Save clustered data to student_cluster_data
        with transaction.atomic():
            with connection.cursor() as cursor:
                for index, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO student_cluster_data (
                            student_id, avg_accuracy, avg_consistency, avg_speed,
                            avg_retention, avg_problem_solving_skills, avg_vocabulary_range,
                            cluster_label
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            avg_accuracy = VALUES(avg_accuracy),
                            avg_consistency = VALUES(avg_consistency),
                            avg_speed = VALUES(avg_speed),
                            avg_retention = VALUES(avg_retention),
                            avg_problem_solving_skills = VALUES(avg_problem_solving_skills),
                            avg_vocabulary_range = VALUES(avg_vocabulary_range),
                            cluster_label = VALUES(cluster_label)
                    """, [
                        row['student_id'],
                        row['avg_accuracy'],
                        row['avg_consistency'],
                        row['avg_speed'],
                        row['avg_retention'],
                        row['avg_problem_solving_skills'],
                        row['avg_vocabulary_range'],
                        row['cluster_label']
                    ])

        return JsonResponse(
            {"message": "Clustering completed and saved to student_cluster_data."},
            status=200
        )

    except Exception:
        print("❌ Full exception during K-means clustering:")
        traceback.print_exc()
        return JsonResponse(
            {"error": "An unexpected error occurred. Check server logs for full details."},
            status=500
        )
