# search/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),  # Map the root URL to index_view
    path('search/', views.search_similar_images, name='search_images'),
]


