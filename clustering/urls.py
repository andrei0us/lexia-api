from django.urls import path
from . import views

urlpatterns = [
    path('algorithm/', views.perform_algorithm),
]
