from django.http import JsonResponse
from django.db import connection, transaction
import pandas as pd
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
                HAVING
                    COUNT(sp.accuracy) > 0 AND
                    COUNT(sp.consistency) > 0 AND
                    COUNT(sp.speed) > 0 AND
                    COUNT(sp.retention) > 0 AND
                    COUNT(sp.problem_solving_skills) > 0 AND
                    COUNT(sp.vocabulary_range) > 0
            """)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            return JsonResponse({"error": "No data found."}, status=400)

        # 2. Label students based on average score thresholds
        def compute_label(row):
            avg_score = row[['avg_accuracy', 'avg_consistency', 'avg_speed', 'avg_retention',
                             'avg_problem_solving_skills', 'avg_vocabulary_range']].mean()
            if avg_score >= 8.5:
                return 'High Achiever'
            elif avg_score >= 6.5:
                return 'Developing Learner'
            else:
                return 'Needs Support'

        df['cluster_label'] = df.apply(compute_label, axis=1)

        # 3. Save results
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

        # 4. Subject averages
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

        subject_averages = {"english": {}, "science": {}}
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

        # 4b. Section + Subject breakdowns
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
            section_id = str(row['fk_section_id'])
            subject_key = 'english' if row['fk_subject_id'] == 1 else 'science'
            if section_id not in section_subject_averages:
                section_subject_averages[section_id] = {}

            section_subject_averages[section_id][subject_key] = {
                "avg_accuracy": row["avg_accuracy"],
                "avg_consistency": row["avg_consistency"],
                "avg_speed": row["avg_speed"],
                "avg_retention": row["avg_retention"],
                "avg_problem_solving_skills": row["avg_problem_solving_skills"],
                "avg_vocabulary_range": row["avg_vocabulary_range"]
            }

        # 5. Return final response
        return JsonResponse({
            "message": "Clustering completed and saved to student_cluster_data.",
            "clustered_data": df.to_dict(orient='records'),
            "subject_averages": subject_averages,
            "section_subject_averages": section_subject_averages
        }, status=200)

    except Exception:
        print("❌ Full exception during clustering:")
        traceback.print_exc()
        return JsonResponse(
            {"error": "Unexpected error occurred. Check logs."},
            status=500
        )
