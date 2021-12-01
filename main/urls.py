from django.contrib import admin
from django.urls import path,include
from . import views
from django.conf.urls.static import static
from website.settings import DEBUG,MEDIA_URL,MEDIA_ROOT
from rest_framework import routers

router= routers.DefaultRouter()

urlpatterns = [
    path("model",views.index,name="main"),
    path("upload_image",views.upload)
]+static(MEDIA_URL,document_root=MEDIA_ROOT)