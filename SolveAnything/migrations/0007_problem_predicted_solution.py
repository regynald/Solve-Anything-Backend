# -*- coding: utf-8 -*-
# Generated by Django 1.9 on 2015-12-10 08:31
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SolveAnything', '0006_auto_20151206_2311'),
    ]

    operations = [
        migrations.AddField(
            model_name='problem',
            name='predicted_solution',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
