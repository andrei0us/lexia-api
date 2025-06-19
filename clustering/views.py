from django.http import JsonResponse
from django.db import connection, transaction
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import traceback

@csrf_exempt
def perform_algorithm(request):
    try:
        # 1. Fetch overall student performance data
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

        # 2. Preprocess
        perf_cols = [
            'avg_accuracy', 'avg_consistency', 'avg_speed',
            'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range'
        ]
        df.dropna(subset=perf_cols, inplace=True)
        df[perf_cols] = df[perf_cols].astype(float)

        # 3. Compute current performance category using quantiles
        df['overall_performance_score'] = df[perf_cols].mean(axis=1)
        q1 = df['overall_performance_score'].quantile(0.33)
        q2 = df['overall_performance_score'].quantile(0.67)
        bins = [df['overall_performance_score'].min() - 0.01, q1, q2, df['overall_performance_score'].max() + 0.01]
        labels = ['Low Performance', 'Medium Performance', 'High Performance']
        df['overall_performance_category'] = pd.cut(df['overall_performance_score'], bins=bins, labels=labels)

        # 4. Encode labels for model training
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['overall_performance_category'])

        # 5. Train/test split
        X = df[perf_cols]
        y = df['label_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Train model using best hyperparameters (pre-tuned)
        model = RandomForestClassifier(
            n_estimators=75,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            criterion='entropy',
            random_state=42
        )
        model.fit(X_train, y_train)

        # 7. Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # 8. Predict live
        df['predicted_label'] = model.predict(X)
        df['predicted_category'] = le.inverse_transform(df['predicted_label'])

        # 9. Save to DB — exclude pred_performance, rename overall_performance_category to cluster_label
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
                        row['overall_performance_category']
                    ])

        # 10. Subject averages
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

        # 11. Section + Subject breakdowns
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

        # 12. Final response
        clustered_data = []
        for _, row in df.iterrows():
            clustered_data.append({
                "student_id": row['student_id'],
                "fk_section_id": row['fk_section_id'],
                "avg_accuracy": row['avg_accuracy'],
                "avg_consistency": row['avg_consistency'],
                "avg_speed": row['avg_speed'],
                "avg_retention": row['avg_retention'],
                "avg_problem_solving_skills": row['avg_problem_solving_skills'],
                "avg_vocabulary_range": row['avg_vocabulary_range'],
                "overall_performance_score": row['overall_performance_score'],
                "overall_performance_category": row['overall_performance_category'],
                "encoded_category": row['label_encoded'],
                "pred_performance": row['predicted_category']
            })

        return JsonResponse({
            "message": "Random Forest prediction and clustering complete.",
            "model_accuracy": accuracy,
            "classification_report": class_report,
            "clustered_data": clustered_data,
            "subject_averages": subject_averages,
            "section_subject_averages": section_subject_averages
        }, status=200)

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": f"Unexpected error occurred: {str(e)}"}, status=500)
