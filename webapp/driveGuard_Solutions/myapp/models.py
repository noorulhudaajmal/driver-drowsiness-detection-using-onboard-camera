from django.db import models
from django.contrib.auth.models import User
from tensorflow.keras.models import load_model
import os

# Create your models here.
class Trained_Model(models.Model):
    sno = models.AutoField(primary_key=True)
    blink_model = models.FileField(upload_to='models/')
    yawning_model = models.FileField(upload_to='models/')
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Model Released on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}'

class Complaint(models.Model):
    complaintID = models.AutoField(primary_key=True)
    footage = models.FileField(upload_to='footages/')
    comments = models.TextField(max_length=500)
    response = models.TextField(max_length=500)
    status = models.TextField(max_length=500)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.complaintID}. By {self.user.username}"


class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    description = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.title} for {self.user.username} at {self.timestamp}"
