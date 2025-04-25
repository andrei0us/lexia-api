from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.db import connection
import pandas as pd
from sklearn.cluster import KMeans

def perform_kmeans(request):
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT accuracy, consistency, speed, retention, problem_solving_skills, vocabulary_range
            FROM students_progress_tbl
        """)
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=[
        "accuracy", "consistency", "speed", "retention", "problem_solving_skills", "vocabulary_range"
    ])

    # 🧼 Drop rows that have NaN values
    df.dropna(inplace=True)

    # 🚫 Prevent running clustering on too little data
    if len(df) < 3:
        return JsonResponse({"error": "Not enough complete data for clustering."}, status=400)

    # 🧠 Run K-means
    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(df)

    return JsonResponse(df.to_dict(orient='records'), safe=False)
