from django.http import JsonResponse
from django.db import connection, transaction
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import traceback

def perform_kmeans(request):
    try:
        # 1. Fetch aggregated per-student performance
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    sp.student_id,
                    st.fk_section_id,
                    AVG(sp.accuracy) AS avg_accuracy,
                    AVG(sp.consistency) AS avg_consistency,
                    AVG(sp.speed) AS avg_speed,
                    AVG(sp.retention) AS avg_retention,
                    AVG(sp.problem_solving_skills) AS avg_problem_solving_skills,
                    AVG(sp.vocabulary_range) AS avg_vocabulary_range
                FROM students_progress_tbl sp
                JOIN students_tbl st ON sp.student_id = st.student_id
                GROUP BY sp.student_id, st.fk_section_id
            """)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=columns)
        df.dropna(inplace=True)  # ✅ Remove students with incomplete data

        if df.shape[0] < 3:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM student_cluster_data")
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
            df_existing = pd.DataFrame(rows, columns=columns)
            return JsonResponse(df_existing.to_dict(orient="records"), safe=False)

        # 2. Prepare features and scale them
        features = df[['avg_accuracy', 'avg_consistency', 'avg_speed',
                       'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # 3. Run KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(scaled_features)
        df['cluster'] = cluster_labels

        # 4. Assign label names based on actual average cluster performance
        cluster_performance = df.groupby('cluster')[[
            'avg_accuracy', 'avg_consistency', 'avg_speed',
            'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range'
        ]].mean().mean(axis=1)  # Mean of all metrics per cluster

        sorted_clusters = cluster_performance.sort_values(ascending=False).index.tolist()
        label_map = {
            sorted_clusters[0]: 'High Achiever',
            sorted_clusters[1]: 'Developing Learner',
            sorted_clusters[2]: 'Needs Support'
        }
        df['cluster_label'] = df['cluster'].map(label_map)

        # 5. Save to database
        with transaction.atomic():
            with connection.cursor() as cursor:
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO student_cluster_data (
                            student_id, fk_section_id,
                            avg_accuracy, avg_consistency, avg_speed,
                            avg_retention, avg_problem_solving_skills, avg_vocabulary_range,
                            cluster_label
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            fk_section_id = VALUES(fk_section_id),
                            avg_accuracy = VALUES(avg_accuracy),
                            avg_consistency = VALUES(avg_consistency),
                            avg_speed = VALUES(avg_speed),
                            avg_retention = VALUES(avg_retention),
                            avg_problem_solving_skills = VALUES(avg_problem_solving_skills),
                            avg_vocabulary_range = VALUES(avg_vocabulary_range),
                            cluster_label = VALUES(cluster_label)
                    """, [
                        row['student_id'],
                        row['fk_section_id'],
                        row['avg_accuracy'],
                        row['avg_consistency'],
                        row['avg_speed'],
                        row['avg_retention'],
                        row['avg_problem_solving_skills'],
                        row['avg_vocabulary_range'],
                        row['cluster_label']
                    ])

        # 6. Subject averages
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
            data = {
                "avg_accuracy": row["avg_accuracy"],
                "avg_consistency": row["avg_consistency"],
                "avg_speed": row["avg_speed"],
                "avg_retention": row["avg_retention"],
                "avg_problem_solving_skills": row["avg_problem_solving_skills"],
                "avg_vocabulary_range": row["avg_vocabulary_range"]
            }
            if row["fk_subject_id"] == 1:
                subject_averages["english"] = data
            elif row["fk_subject_id"] == 2:
                subject_averages["science"] = data

        # 7. Section + subject breakdown
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    s.fk_section_id,
                    sp.fk_subject_id,
                    AVG(sp.accuracy) AS avg_accuracy,
                    AVG(sp.consistency) AS avg_consistency,
                    AVG(sp.speed) AS avg_speed,
                    AVG(sp.retention) AS avg_retention,
                    AVG(sp.problem_solving_skills) AS avg_problem_solving_skills,
                    AVG(sp.vocabulary_range) AS avg_vocabulary_range
                FROM students_progress_tbl sp
                JOIN students_tbl s ON s.student_id = sp.student_id
                GROUP BY s.fk_section_id, sp.fk_subject_id
            """)
            section_rows = cursor.fetchall()
            section_columns = [col[0] for col in cursor.description]
            df_sections = pd.DataFrame(section_rows, columns=section_columns)

        section_subject_averages = {}
        for _, row in df_sections.iterrows():
            sec_id = str(row['fk_section_id'])
            subj = 'english' if row['fk_subject_id'] == 1 else 'science'
            if sec_id not in section_subject_averages:
                section_subject_averages[sec_id] = {}
            section_subject_averages[sec_id][subj] = {
                "avg_accuracy": row["avg_accuracy"],
                "avg_consistency": row["avg_consistency"],
                "avg_speed": row["avg_speed"],
                "avg_retention": row["avg_retention"],
                "avg_problem_solving_skills": row["avg_problem_solving_skills"],
                "avg_vocabulary_range": row["avg_vocabulary_range"]
            }

        return JsonResponse({
            "message": "✅ Clustering complete",
            "clustered_data": df.to_dict(orient='records'),
            "subject_averages": subject_averages,
            "section_subject_averages": section_subject_averages
        }, status=200)

    except Exception:
        print("❌ Error during KMeans clustering:")
        traceback.print_exc()
        return JsonResponse(
            {"error": "Unexpected error occurred. Check logs."},
            status=500
        )
