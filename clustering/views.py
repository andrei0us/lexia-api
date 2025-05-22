from django.http import JsonResponse
from django.db import connection
import pandas as pd
from sklearn.cluster import KMeans
from django.db import transaction # Import transaction for atomicity

def perform_kmeans(request):
    # This view is intended to be triggered by a backend process (e.g., a cron job),
    # NOT directly by a user's browser request for immediate analytics display.
    # If a user hits this directly, it will run the K-means calculation.

    try:
        with connection.cursor() as cursor:
            # 1. Fetch aggregated performance metrics along with student_id
            # This query calculates the average performance for each student across all their recorded activities.
            # You might need to adjust the time window for averaging if required (e.g., last 30 days).
            cursor.execute("""
                SELECT
                    spr.student_id,
                    AVG(spr.accuracy) AS avg_accuracy,
                    AVG(spr.consistency) AS avg_consistency,
                    AVG(spr.speed) AS avg_speed,
                    AVG(spr.retention) AS avg_retention,
                    AVG(spr.problem_solving_skills) AS avg_problem_solving_skills,
                    AVG(spr.vocabulary_range) AS avg_vocabulary_range
                FROM
                    students_progress_tbl spr
                GROUP BY
                    spr.student_id
                HAVING -- Ensure we only cluster students with complete data for all metrics
                    COUNT(spr.accuracy) > 0 AND
                    COUNT(spr.consistency) > 0 AND
                    COUNT(spr.speed) > 0 AND
                    COUNT(spr.retention) > 0 AND
                    COUNT(spr.problem_solving_skills) > 0 AND
                    COUNT(spr.vocabulary_range) > 0
            """)
            # Store student_ids separately to map clusters back
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)

        # Ensure that 'student_id' column is treated as an identifier, not a feature for clustering
        student_ids = df['student_id']
        features_df = df.drop(columns=['student_id'])

        # 🚫 Prevent running clustering on too little data
        if len(features_df) < 3:
            return JsonResponse({"message": "Not enough complete data for clustering. Skipping K-means execution."}, status=200) # Return 200 as it's not an error, just no data

        # 🧼 Drop rows that have NaN values (though AVG should handle this, good for robustness)
        features_df.dropna(inplace=True)

        # Re-check data after dropping NaNs
        if len(features_df) < 3:
            return JsonResponse({"message": "Not enough complete data for clustering after NaN removal. Skipping K-means execution."}, status=200)

        # 🧠 Run K-means
        # It's a good practice to set random_state for reproducibility
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto') # n_init='auto' or provide a value for newer sklearn versions
        cluster_labels = kmeans.fit_predict(features_df)

        # Map cluster labels back to student_ids
        df['cluster_label'] = pd.Series(cluster_labels, index=features_df.index).map({
            0: 'Cluster A', # Assign meaningful names
            1: 'Cluster B',
            2: 'Cluster C'
        })
        # If you want to use the raw cluster numbers, just assign `cluster_labels` directly.

        # 2. Save the clustered data into `student_cluster_data` table
        # Use a transaction to ensure all updates are atomic
        with transaction.atomic():
            with connection.cursor() as cursor:
                # Iterate through the DataFrame to update/insert into student_cluster_data
                for index, row in df.iterrows():
                    # Use ON DUPLICATE KEY UPDATE for MySQL or UPSERT for PostgreSQL/SQLite
                    # Assuming MySQL for this example
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
                            avg_problem_solving_skills = VALUES(problem_solving_skills),
                            avg_vocabulary_range = VALUES(vocabulary_range),
                            cluster_label = VALUES(cluster_label)
                    """, [
                        row['student_id'], row['avg_accuracy'], row['avg_consistency'],
                        row['avg_speed'], row['avg_retention'], row['avg_problem_solving_skills'],
                        row['avg_vocabulary_range'], row['cluster_label']
                    ])

        return JsonResponse({"message": "K-means clustering completed and student_cluster_data updated successfully."}, status=200)

    except Exception as e:
        # Log the error for debugging
        print(f"Error during K-means execution: {e}")
        return JsonResponse({"error": f"An error occurred during K-means processing: {str(e)}"}, status=500)