from django.http import JsonResponse
from django.db import connection, transaction
import pandas as pd
from sklearn.cluster import KMeans
import traceback

def perform_kmeans(request):
    try:
        # 1. Fetch aggregated per-student performance
        with connection.cursor() as cursor:
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
        student_ids = df['student_id']
        features_df = df.drop(columns=['student_id'])
        features_df.dropna(inplace=True)

        if len(features_df) < 3:
            return JsonResponse({"message": "Not enough data for clustering."}, status=200)

        # 2. Run KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(features_df)

        label_map = {
            0: 'High Achiever',
            1: 'Developing Learner',
            2: 'Needs Support'
        }

        df['cluster_label'] = pd.Series(cluster_labels, index=features_df.index).map(label_map)

        # 3. Save results to student_cluster_data
        with transaction.atomic():
            with connection.cursor() as cursor:
                for _, row in df.iterrows():
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

        # 4. Also calculate average metrics per subject (English vs. Science)
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    fk_subject_id,
                    AVG(accuracy) AS avg_accuracy,
                    AVG(consistency) AS avg_consistency,
                    AVG(speed) AS avg_speed,
                    AVG(retention) AS avg_retention,
                    AVG(problem_solving_skills) AS avg_problem_solving_skills,
                    AVG(vocabulary_range) AS avg_vocabulary_range
                FROM students_progress_tbl
                GROUP BY fk_subject_id
            """)
            subject_rows = cursor.fetchall()
            subject_columns = [col[0] for col in cursor.description]
            df_subjects = pd.DataFrame(subject_rows, columns=subject_columns)

        subject_averages = {
            "english": {},
            "science": {}
        }

        for _, row in df_subjects.iterrows():
            subject_data = {
                "avg_accuracy": row["avg_accuracy"],
                "avg_consistency": row["avg_consistency"],
                "avg_speed": row["avg_speed"],
                "avg_retention": row["avg_retention"],
                "avg_problem_solving_skills": row["avg_problem_solving_skills"],
                "avg_vocabulary_range": row["avg_vocabulary_range"]
            }

            if row["fk_subject_id"] == 1:
                subject_averages["english"] = subject_data
            elif row["fk_subject_id"] == 2:
                subject_averages["science"] = subject_data

        # 5. Return all data as JSON for frontend
        return JsonResponse({
            "message": "Clustering completed and saved to student_cluster_data.",
            "clustered_data": df.to_dict(orient='records'),
            "subject_averages": subject_averages
        }, status=200)

    except Exception:
        print("❌ Full exception during clustering:")
        traceback.print_exc()
        return JsonResponse(
            {"error": "Unexpected error occurred. Check logs."},
            status=500
        )
