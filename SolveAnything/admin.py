from django.contrib import admin
from SolveAnything.models import Problem

# Register your models here.
class ProblemAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_date', 'image', 'processed_image', 'classification',
                    'predicted_solution', 'solution', 'correct',)
    list_filter = ('created_date', 'correct',)
    search_fields = ('id', 'created_date', 'classification', 'predicted_solution', 'solution', 'correct',)
    ordering = ('-created_date',)

admin.site.register(Problem, ProblemAdmin)