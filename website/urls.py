from django.contrib import admin
from django.urls import path, include
from website import views

urlpatterns = [
    path("", views.index, name='home'),
    path("livecam_feed", views.livecam_feed, name="livecam_feed")
]