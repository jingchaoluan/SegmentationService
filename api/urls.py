from rest_framework.urlpatterns import format_suffix_patterns
from django.conf.urls import url
from api import views

urlpatterns = [
    url(r'^$', views.segmentationView, name='segmentationView'),
]

urlpatterns = format_suffix_patterns(urlpatterns)
