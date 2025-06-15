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
                           SELECT sp.student_id,
                                  st.fk_section_id,
                                  AVG(sp.accuracy)               AS avg_accuracy,
                                  AVG(sp.consistency)            AS avg_consistency,
                                  AVG(sp.speed)                  AS avg_speed,
                                  AVG(sp.retention)              AS avg_retention,
                                  AVG(sp.problem_solving_skills) AS avg_problem_solving_skills,
                                  AVG(sp.vocabulary_range)       AS avg_vocabulary_range,
                                  COUNT(*)                       AS record_count
                           FROM students_progress_tbl sp
                                    JOIN students_tbl st ON sp.student_id = st.student_id
                           WHERE sp.accuracy IS NOT NULL
                             AND sp.consistency IS NOT NULL
                             AND sp.speed IS NOT NULL
                             AND sp.retention IS NOT NULL
                             AND sp.problem_solving_skills IS NOT NULL
                             AND sp.vocabulary_range IS NOT NULL
                           GROUP BY sp.student_id, st.fk_section_id
                           HAVING COUNT(*) >= 3
                           """)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=columns)

        # Additional data quality checks
        numeric_cols = ['avg_accuracy', 'avg_consistency', 'avg_speed',
                        'avg_retention', 'avg_problem_solving_skills', 'avg_vocabulary_range']

        # Convert Decimal to float to avoid type issues
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with any null values or zeros in critical metrics
        df = df.dropna(subset=numeric_cols)
        df = df[(df[numeric_cols] > 0).all(axis=1)]

        print(f"📊 Students available for clustering: {len(df)}")

        if df.shape[0] < 6:  # Need at least 6 students for meaningful 3-cluster analysis
            # Clear stale data and return empty result
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM student_cluster_data")

            return JsonResponse({
                "message": "⚠️ Insufficient data for clustering (need at least 6 students with complete records)",
                "clustered_data": [],
                "subject_averages": {},
                "section_subject_averages": {}
            }, status=200)

        # 2. Prepare features and scale them
        features = df[numeric_cols].copy().astype(float)  # Ensure all are float type

        # Remove outliers using IQR method to improve clustering
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # 3. Determine optimal number of clusters (2 or 3 based on data size)
        n_clusters = min(3, max(2, len(df) // 3))
        print(f"🎯 Using {n_clusters} clusters for {len(df)} students")

        # Run KMeans clustering with multiple attempts for stability
        best_kmeans = None
        best_inertia = float('inf')

        for _ in range(5):  # Try 5 times to get the best clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=None, n_init=10)
            kmeans.fit(scaled_features)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans

        cluster_labels = best_kmeans.labels_
        df['cluster'] = cluster_labels

        # 4. Create comprehensive performance score for ranking
        # Weight important metrics more heavily
        weights = {
            'avg_accuracy': 0.25,
            'avg_consistency': 0.20,
            'avg_speed': 0.15,
            'avg_retention': 0.20,
            'avg_problem_solving_skills': 0.15,
            'avg_vocabulary_range': 0.05
        }

        # Calculate weighted composite score for each student
        df['composite_score'] = 0.0
        for col, weight in weights.items():
            df['composite_score'] += df[col].astype(float) * weight

        # Calculate cluster performance using composite scores
        cluster_stats = df.groupby('cluster').agg({
            'composite_score': ['mean', 'median', 'std', 'count'],
            **{col: 'mean' for col in numeric_cols}
        }).round(3)

        # Rank clusters by their median composite score (more robust than mean)
        cluster_performance = df.groupby('cluster')['composite_score'].median().sort_values(ascending=False)

        print("📈 Cluster Performance Ranking:")
        for rank, (cluster_id, score) in enumerate(cluster_performance.items()):
            student_count = len(df[df['cluster'] == cluster_id])
            print(f"  Rank {rank + 1}: Cluster {cluster_id} - Score: {score:.3f} ({student_count} students)")

        # 5. Assign meaningful labels based on performance ranking
        cluster_ranks = list(cluster_performance.index)

        if n_clusters == 3:
            label_map = {
                cluster_ranks[0]: 'High Achiever',
                cluster_ranks[1]: 'Developing Learner',
                cluster_ranks[2]: 'Needs Support'
            }
        else:  # n_clusters == 2
            label_map = {
                cluster_ranks[0]: 'High Achiever',
                cluster_ranks[1]: 'Developing Learner'
            }

        df['cluster_label'] = df['cluster'].map(label_map)

        # Validation: Check if labeling makes sense
        print("🔍 Validation Check:")
        for label in df['cluster_label'].unique():
            subset = df[df['cluster_label'] == label]
            avg_score = subset['composite_score'].mean()
            print(f"  {label}: {len(subset)} students, Avg Score: {avg_score:.3f}")

        # 6. Clear old data and save new clustering results
        with transaction.atomic():
            # Clear existing cluster data
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM student_cluster_data")

            # Insert new clustering results
            with connection.cursor() as cursor:
                for _, row in df.iterrows():
                    cursor.execute("""
                                   INSERT INTO student_cluster_data (student_id, fk_section_id,
                                                                     avg_accuracy, avg_consistency, avg_speed,
                                                                     avg_retention, avg_problem_solving_skills,
                                                                     avg_vocabulary_range,
                                                                     cluster_label)
                                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                   """, [
                                       int(row['student_id']),
                                       int(row['fk_section_id']),
                                       float(row['avg_accuracy']),
                                       float(row['avg_consistency']),
                                       float(row['avg_speed']),
                                       float(row['avg_retention']),
                                       float(row['avg_problem_solving_skills']),
                                       float(row['avg_vocabulary_range']),
                                       str(row['cluster_label'])
                                   ])

        # 7. Calculate subject averages (only from students with complete data)
        with connection.cursor() as cursor:
            cursor.execute("""
                           SELECT fk_subject_id,
                                  AVG(accuracy)               AS avg_accuracy,
                                  AVG(consistency)            AS avg_consistency,
                                  AVG(speed)                  AS avg_speed,
                                  AVG(retention)              AS avg_retention,
                                  AVG(problem_solving_skills) AS avg_problem_solving_skills,
                                  AVG(vocabulary_range)       AS avg_vocabulary_range
                           FROM students_progress_tbl
                           WHERE accuracy IS NOT NULL
                             AND consistency IS NOT NULL
                             AND speed IS NOT NULL
                             AND retention IS NOT NULL
                             AND problem_solving_skills IS NOT NULL
                             AND vocabulary_range IS NOT NULL
                             AND accuracy > 0
                             AND consistency > 0
                             AND speed > 0
                             AND retention > 0
                             AND problem_solving_skills > 0
                             AND vocabulary_range > 0
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
                "avg_accuracy": round(float(row["avg_accuracy"]), 3),
                "avg_consistency": round(float(row["avg_consistency"]), 3),
                "avg_speed": round(float(row["avg_speed"]), 3),
                "avg_retention": round(float(row["avg_retention"]), 3),
                "avg_problem_solving_skills": round(float(row["avg_problem_solving_skills"]), 3),
                "avg_vocabulary_range": round(float(row["avg_vocabulary_range"]), 3)
            }
            if row["fk_subject_id"] == 1:
                subject_averages["english"] = data
            elif row["fk_subject_id"] == 2:
                subject_averages["science"] = data

        # 8. Section + subject breakdown (with data quality filters)
        with connection.cursor() as cursor:
            cursor.execute("""
                           SELECT s.fk_section_id,
                                  sp.fk_subject_id,
                                  AVG(sp.accuracy)               AS avg_accuracy,
                                  AVG(sp.consistency)            AS avg_consistency,
                                  AVG(sp.speed)                  AS avg_speed,
                                  AVG(sp.retention)              AS avg_retention,
                                  AVG(sp.problem_solving_skills) AS avg_problem_solving_skills,
                                  AVG(sp.vocabulary_range)       AS avg_vocabulary_range
                           FROM students_progress_tbl sp
                                    JOIN students_tbl s ON s.student_id = sp.student_id
                           WHERE sp.accuracy IS NOT NULL
                             AND sp.consistency IS NOT NULL
                             AND sp.speed IS NOT NULL
                             AND sp.retention IS NOT NULL
                             AND sp.problem_solving_skills IS NOT NULL
                             AND sp.vocabulary_range IS NOT NULL
                             AND sp.accuracy > 0
                             AND sp.consistency > 0
                             AND sp.speed > 0
                             AND sp.retention > 0
                             AND sp.problem_solving_skills > 0
                             AND sp.vocabulary_range > 0
                           GROUP BY s.fk_section_id, sp.fk_subject_id
                           HAVING COUNT(*) >= 2
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
                "avg_accuracy": round(float(row["avg_accuracy"]), 3),
                "avg_consistency": round(float(row["avg_consistency"]), 3),
                "avg_speed": round(float(row["avg_speed"]), 3),
                "avg_retention": round(float(row["avg_retention"]), 3),
                "avg_problem_solving_skills": round(float(row["avg_problem_solving_skills"]), 3),
                "avg_vocabulary_range": round(float(row["avg_vocabulary_range"]), 3)
            }

        # Prepare final response with rounded values
        clustered_data = df.drop(columns=['record_count', 'composite_score']).round(3).to_dict(orient='records')

        return JsonResponse({
            "message": f"✅ Clustering complete with {n_clusters} clusters",
            "total_students_clustered": len(df),
            "clustered_data": clustered_data,
            "subject_averages": subject_averages,
            "section_subject_averages": section_subject_averages,
            "cluster_summary": {
                label: len(df[df['cluster_label'] == label])
                for label in df['cluster_label'].unique()
            }
        }, status=200)

    except Exception as e:
        print("❌ Error during KMeans clustering:")
        traceback.print_exc()
        return JsonResponse(
            {"error": f"Clustering failed: {str(e)}"},
            status=500
        )