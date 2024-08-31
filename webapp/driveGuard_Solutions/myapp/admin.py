from django.contrib import admin
from .models import Trained_Model,Complaint, Notification


admin.site.register(Trained_Model)
admin.site.register(Complaint)
admin.site.register(Notification)