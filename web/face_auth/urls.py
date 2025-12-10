"""
URL configuration for face_auth app
"""
from django.urls import path
from . import views

app_name = 'face_auth'

urlpatterns = [
    # Main dual authentication page (default)
    path('', views.dual_auth_page, name='dual_auth'),
    
    # Individual authentication pages
    path('face/', views.face_page, name='face_page'),
    path('voice/', views.voice_page, name='voice_page'),
    
    # API endpoints
    path('api/predict/', views.predict_face, name='predict_face'),
    path('api/predict_voice/', views.predict_voice, name='predict_voice'),
    path('api/authenticate_dual/', views.authenticate_dual, name='authenticate_dual'),
]
