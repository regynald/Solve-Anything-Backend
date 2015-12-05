from django.contrib import admin
from SolveAnything.models import Problem

# Register your models here.
class ProblemAdmin(admin.ModelAdmin):
    pass

admin.site.register(Problem, ProblemAdmin)