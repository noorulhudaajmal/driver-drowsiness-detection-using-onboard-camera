# Generated by Django 5.1 on 2024-08-30 20:03

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0003_rename_complaints_complaint_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RemoveField(
            model_name='complaint',
            name='username',
        ),
        migrations.AddField(
            model_name='complaint',
            name='response',
            field=models.TextField(default='UnResolved', max_length=500),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='complaint',
            name='status',
            field=models.TextField(default=None, max_length=500),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='complaint',
            name='user',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
    ]