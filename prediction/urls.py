import django
from django.urls import include, path
from .views import *


urlpatterns = [
    path('index', index,name="index"),
]
