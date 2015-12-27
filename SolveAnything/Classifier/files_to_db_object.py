import os

from django.core.files import File
from SolveAnything.models import Problem

root_dir = ''#'/Users/regynald/Desktop/Operators/PNGs'

for directory, subdirectories, files in os.walk(root_dir):
    for file in files:
      if file != '.DS_Store':
          reopen = open(os.path.join(directory, file))
          django_file = File(reopen)
          p = Problem(image=django_file)
          p.save()