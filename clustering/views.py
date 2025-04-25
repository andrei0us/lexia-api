from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from sklearn.cluster import KMeans
import pandas as pd
import MySQLdb
from django.db import connection
import pandas as pd
from sklearn.cluster import KMeans
from django.http import JsonResponse


def perform_kmeans(request):
    with connection.cursor() as cursor:
        # Adjust this to match your actual table and columns in Hostinger
        cursor.execute("SELECT accuracy, consistency, speed, retention, problem_solving_skills, vocabulary_range FROM student_performance")
        rows = cursor.fetchall()

    # Convert data to DataFrame
    df = pd.DataFrame(rows, columns=[
        "accuracy", "consistency", "speed", "retention", "problem_solving_skills", "vocabulary_range"
    ])

    if len(df) < 3:
        return JsonResponse({"error": "Not enough data for clustering"}, status=400)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(df)

    return JsonResponse(df.to_dict(orient='records'), safe=False)

