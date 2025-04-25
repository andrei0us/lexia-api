from django.contrib import admin
from django.urls import path, include  # ✅ include is used to connect app URLs

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('clustering.urls')),  # ✅ this loads routes from clustering/urls.py
]
