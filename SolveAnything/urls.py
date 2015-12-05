from django.conf.urls import url
from SolveAnything import views

urlpatterns = [
    url(r'^classify/$', views.classify),
    url(r'^label/$', views.label),
    url(r'^problems/$', views.problem_list),
]