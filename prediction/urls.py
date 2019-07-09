import django
from django.urls import include, path
from .views import *


urlpatterns = [
    path('', index,name="index"),
    path('esp/', esp, name="esp")
]
