from django.shortcuts import render

# Create your views here.

from SolveAnything.models import Problem
from SolveAnything.serializers import ProblemSerializer
from Classifier.classify_problem import classify_problem

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['POST'])
def classify(request):
    """
    Classifies a handwritten math problem. Returns string of classified solution, and id of database entry.
    """
    if request.method == 'POST':
        serializer = ProblemSerializer(data=request.data)
        if serializer.is_valid():
            problem = serializer.save()
            problem.save()
            # THIS IS SAMPLE ONLY. PLEASE DON'T USE
            classification, predicted_solution = classify_problem(problem.id)
            problem.refresh_from_db()
            problem.classification = classification
            problem.predicted_solution = predicted_solution
            problem.save()

            response_data = {
                'id': problem.id,
                'classification': problem.classification,
                'predicted_solution': problem.predicted_solution,
                'processed_image': problem.processed_image.url
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def label(request):
    """
    Labels a classified problem as correct, or incorrect. If incorrect adds correct solution to object.
    """
    if request.method == 'POST':
        request_data = request.data
        problem = Problem.objects.get(pk=request_data['id'])
        problem.correct = request_data['correct']
        if 'solution' in request_data:
            problem.solution = request_data['solution']
        else:
            problem.solution = problem.classification
        problem.save()
        return Response(status=status.HTTP_202_ACCEPTED)

@api_view(['GET'])
def problem_list(request):
    """
    List all problems stored in database
    """
    if request.method == 'GET':
        problems = Problem.objects.all()
        serializer = ProblemSerializer(problems, many=True)
        return Response(serializer.data)
