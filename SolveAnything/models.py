from __future__ import unicode_literals

import os
from uuid import uuid4
from django.db import models

def random_name(instance, filename):
    ext = filename.split('.')[-1]
    filename = "%s.%s" % (uuid4(), ext)
    return os.path.join('problem-photos', filename)


class Problem(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)
    image = models.ImageField(upload_to=random_name)
    classification = models.CharField(max_length=100, blank=True, null=True)
    solution = models.CharField(max_length=100, blank=True, null=True)
    correct = models.NullBooleanField(blank=True)