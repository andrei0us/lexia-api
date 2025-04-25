from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from sklearn.cluster import KMeans
import pandas as pd
import MySQLdb

def perform_kmeans(request):
    db = MySQLdb.connect("localhost", "root", "", "mydb")
    cursor = db.cursor()

    # Example query, update based on your table
    cursor.execute("SELECT accuracy, consistency, speed, retention, problem_solving_skills, vocabulary_range FROM students_progress_tbl")
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=["accuracy", "consistency", "speed", "retention", "problem_solving_skills",
                                     "vocabulary_range"])

    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(df)

    return JsonResponse(df.to_dict(orient='records'), safe=False)

