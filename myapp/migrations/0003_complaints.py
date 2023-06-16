# Generated by Django 4.1.5 on 2023-04-06 21:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("myapp", "0002_cnn_models_timestamp"),
    ]

    operations = [
        migrations.CreateModel(
            name="Complaints",
            fields=[
                ("complaintID", models.AutoField(primary_key=True, serialize=False)),
                ("footage", models.FilePathField()),
                ("comments", models.TextField(max_length=500)),
                ("username", models.TextField(max_length=20)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
