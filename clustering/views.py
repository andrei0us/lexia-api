from django.http import JsonResponse
from django.db import connection
import pandas as pd
from sklearn.cluster import KMeans
from django.db import transaction


def perform_kmeans(request):
    """
    This Django view performs K-means clustering on aggregated student performance
    from 'students_progress_tbl' and updates/inserts results into 'student_cluster_data' table.

    This should be triggered by a backend cron job/scheduler, not directly by a user.
    """
    try:
        with connection.cursor() as cursor:
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
                HAVING -- Ensure we only cluster students with sufficient data for all metrics
                    COUNT(accuracy) > 0 AND
                    COUNT(consistency) > 0 AND
                    COUNT(speed) > 0 AND
                    COUNT(retention) > 0 AND
                    COUNT(problem_solving_skills) > 0 AND
                    COUNT(vocabulary_range) > 0
            """)

            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        # Convert to DataFrame
        df_aggregated = pd.DataFrame(rows, columns=columns)

        # Separate student_id from the features for clustering
        student_ids_for_clustering = df_aggregated['student_id']
        features_df = df_aggregated.drop(columns=['student_id'])

        # 🚫 Prevent running clustering on too little data
        if len(features_df) < 3:
            return JsonResponse(
                {"message": "Not enough complete aggregated data for clustering. Skipping K-means execution."},
                status=200
            )

        # 🧼 Drop rows from features_df that might have NaN values (e.g., if AVG returned NULL for some reason)
        features_df.dropna(inplace=True)

        # Re-check data after dropping NaNs, as dropna might remove too many rows
        if len(features_df) < 3:
            return JsonResponse(
                {
                    "message": "Not enough complete aggregated data for clustering after NaN removal. Skipping K-means execution."},
                status=200
            )

        # 🧠 Run K-means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        cluster_labels_raw = kmeans.fit_predict(features_df)

        # Map raw cluster labels to meaningful names (customize these based on your analysis!)
        cluster_label_map = {
            0: 'High Achiever',
            1: 'Developing Learner',
            2: 'Needs Support'
        }

        # Create a DataFrame that includes student_id, aggregated metrics, and the new cluster_label
        # Align the cluster labels with the corresponding student_ids and their aggregated metrics
        df_aggregated['cluster_label'] = pd.Series(cluster_labels_raw, index=features_df.index).map(cluster_label_map)

        # 2. Update/Insert the clustered data into the 'student_cluster_data' table
        with transaction.atomic():
            with connection.cursor() as cursor:
                for index, row in df_aggregated.iterrows():
                    # Assuming 'student_cluster_data' table has 'student_id' as PRIMARY KEY or UNIQUE
                    # This allows ON DUPLICATE KEY UPDATE for MySQL
                    cursor.execute("""
                        INSERT INTO student_cluster_data (
                            student_id, avg_accuracy, avg_consistency, avg_speed,
                            avg_retention, avg_problem_solving_skills, avg_vocabulary_range,
                            cluster_label, last_calculated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON DUPLICATE KEY UPDATE
                            avg_accuracy = VALUES(avg_accuracy),
                            avg_consistency = VALUES(avg_consistency),
                            avg_speed = VALUES(avg_speed),
                            avg_retention = VALUES(avg_retention),
                            avg_problem_solving_skills = VALUES(problem_solving_skills),
                            avg_vocabulary_range = VALUES(vocabulary_range),
                            cluster_label = VALUES(cluster_label),
                            last_calculated_at = NOW()
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
            {"message": "K-means clustering completed and 'student_cluster_data' updated successfully."},
            status=200
        )

    except Exception as e:
        # Log the error for debugging. Use Django's proper logging configuration in production.
        print(f"Error during K-means execution: {e}")
        return JsonResponse(
            {"error": f"An error occurred during K-means processing: {str(e)}"},
            status=500
        )