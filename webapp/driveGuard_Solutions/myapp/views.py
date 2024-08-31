from django.shortcuts import render

from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password

from .models import Trained_Model, Complaint, Notification


# Create your views here.

def home(request):
    return render(request, "home.html")


def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect('user_panel')
        else:
            error = 'Invalid login credentials. Please try again.'
            return render(request, 'home.html', {'error': error})
    return render(request, 'home.html')


@login_required
def user_panel(request):
    user = request.user
    complaints = Complaint.objects.filter(user=user)
    notifications = Notification.objects.filter(user=user).order_by('-timestamp')
    up_models = Trained_Model.objects.all()
    params = {'complaints': complaints, 'up_models': up_models, 'notifications': notifications}
    return render(request, 'user_panel.html', params)


def user_logout(request):
    logout(request)
    return redirect('home')


def register_user(request):
    if request.method == 'POST':
        first_name = request.POST['firstname']
        last_name = request.POST['lastname']
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']

        try:
            validate_password(password)
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )
            user.save()

            messages.success(request, 'Thanks for registering, you will soon receive a confirmation email.')
            return redirect('home')

        except ValidationError as e:
            messages.error(request, f'Password error: {e.messages[0]}')
            return redirect('home')

    return render(request, 'home.html')

@login_required
def submit_complaint(request):
    if request.method == 'POST':
        footage = request.FILES['footage']
        comments = request.POST['comments']

        complaint = Complaint(
            footage=footage,
            comments=comments,
            user=request.user,
            status="Pending",
            response=""
        )
        complaint.save()

        messages.success(request, 'Complaint submitted successfully.')
        return redirect('user_panel')

    return render(request, 'user_panel.html')