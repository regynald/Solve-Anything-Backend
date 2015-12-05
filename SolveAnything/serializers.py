from rest_framework import serializers
from SolveAnything.models import Problem

class ProblemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Problem
        fields = ('id', 'created_date', 'modified_date', 'image', 'classification', 'solution', 'correct')