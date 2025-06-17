from django.http import JsonResponse
from django.db import connection, transaction
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import traceback


@csrf_exempt
def perform_kmeans(request):
    try:
        # 1. Fetch student data
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

        # 2. Label current performance (clustering)
        def compute_cluster_label(row):
            avg_score = row[['avg_accuracy', 'avg_consistency', 'avg_speed',
                             'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range']].mean()
            if avg_score >= 8.5:
                return 'High Performance'
            elif avg_score >= 6.5:
                return 'Medium Performance'
            else:
                return 'Low Performance'

        df['cluster_label'] = df.apply(compute_cluster_label, axis=1)

        # 3. Train and predict future performance using Random Forest
        perf_cols = ['avg_accuracy', 'avg_consistency', 'avg_speed',
                     'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range']
        df['overall_performance_score'] = df[perf_cols].mean(axis=1)

        q1 = df['overall_performance_score'].quantile(0.33)
        q2 = df['overall_performance_score'].quantile(0.67)

        bins = [df['overall_performance_score'].min() - 0.01, q1, q2, df['overall_performance_score'].max() + 0.01]
        labels = ['Low Performance', 'Medium Performance', 'High Performance']
        df['overall_performance_category'] = pd.cut(df['overall_performance_score'], bins=bins, labels=labels)

        le = LabelEncoder()
        df['encoded_category'] = le.fit_transform(df['overall_performance_category'])

        model = RandomForestClassifier(
            criterion='gini',
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=10,
            n_estimators=50,
            random_state=42
        )
        model.fit(df[perf_cols], df['encoded_category'])
        df['pred_performance'] = le.inverse_transform(model.predict(df[perf_cols]))

        # 4. Save results into student_cluster_data
        with transaction.atomic():
            with connection.cursor() as cursor:
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO student_cluster_data (
                            student_id, fk_section_id,
                            avg_accuracy, avg_consistency, avg_speed,
                            avg_retention, avg_problem_solving_skills, avg_vocabulary_range,
                            cluster_label, pred_performance
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            fk_section_id = VALUES(fk_section_id),
                            avg_accuracy = VALUES(avg_accuracy),
                            avg_consistency = VALUES(avg_consistency),
                            avg_speed = VALUES(avg_speed),
                            avg_retention = VALUES(avg_retention),
                            avg_problem_solving_skills = VALUES(avg_problem_solving_skills),
                            avg_vocabulary_range = VALUES(avg_vocabulary_range),
                            cluster_label = VALUES(cluster_label),
                            pred_performance = VALUES(pred_performance)
                    """, [
                        row['student_id'],
                        row['fk_section_id'],
                        row['avg_accuracy'],
                        row['avg_consistency'],
                        row['avg_speed'],
                        row['avg_retention'],
                        row['avg_problem_solving_skills'],
                        row['avg_vocabulary_range'],
                        row['cluster_label'],
                        row['pred_performance']
                    ])

        # 5. Subject averages
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
            df_subjects = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])

        subject_averages = {"english": {}, "science": {}}
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

        # 6. Section + Subject breakdowns
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
            df_sections = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])

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

        # 7. Return
        return JsonResponse({
            "message": "Clustering and prediction complete.",
            "clustered_data": df.to_dict(orient='records'),
            "subject_averages": subject_averages,
            "section_subject_averages": section_subject_averages
        }, status=200)

    except Exception:
        traceback.print_exc()
        return JsonResponse({"error": "Unexpected error occurred. Check logs."}, status=500)
