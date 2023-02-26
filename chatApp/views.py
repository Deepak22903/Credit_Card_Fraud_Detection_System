from django.views.generic import TemplateView
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse 
from chat.forms import *

class HomePage(TemplateView):
	template_name = 'index.html'

def info(request):
		return render(request, 'info.html')


def DETECTION_PAGE(request):   
    return render(request, 'detection.html') 
  

