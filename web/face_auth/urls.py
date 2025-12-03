"""
URL configuration for face_auth app
"""
from django.urls import path
from . import views

app_name = 'face_auth'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/predict/', views.predict_face, name='predict_face'),
]
