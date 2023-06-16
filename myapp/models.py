from django.db import models
# from tensorflow.keras.models import load_model
import os

# Create your models here.
class CNN_Models(models.Model):
    sno = models.AutoField(primary_key=True)
    eye_model = models.FileField(upload_to='models/')
    mouth_model = models.FileField(upload_to='models/')
    timestamp = models.DateTimeField(auto_now_add=True)
        
    def __str__(self):
        return 'Model Released on '+ self.timestamp 
    
class Complaints(models.Model):
    complaintID = models.AutoField(primary_key=True)
    footage = models.FileField(upload_to='footages/')
    comments = models.TextField(max_length=500)
    username = models.TextField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.complaintID+'. By '+ self.username
        
