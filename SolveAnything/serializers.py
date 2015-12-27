from rest_framework import serializers
from SolveAnything.models import Problem

class ProblemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Problem
        fields = ('id', 'created_date', 'modified_date', 'image', 'processed_image', 'classification',
                  'predicted_solution', 'solution', 'correct')