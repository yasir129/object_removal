from django.db import models

# Create your models here.
class uploadImage(models.Model):
    image = models.ImageField("uploadimage")