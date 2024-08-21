from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from myapp.models import CNN_Models, Complaints
from django.contrib import messages
from django.http import JsonResponse


# Create your views here.

def home(request):
    return render(request, "myapp/home.html")


@login_required
@user_passes_test(lambda u: u.is_superuser)
def admin_account(request):
    up_models = CNN_Models.objects.all()
    users = User.objects.all()
    complaints = Complaints.objects.all()
    params = {'up_models': up_models, 'users': users, 'complaints': complaints}
    return render(request, 'myapp/admin_account.html', params)


@login_required
def user_account(request):
    up_models = CNN_Models.objects.all()
    params = {'up_models': up_models}
    return render(request, 'myapp/user_account.html', params)


def register_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password1 = request.POST['password']
        password2 = request.POST['password2']
        email = request.POST['email']
        fname = request.POST['first_name']
        lname = request.POST['last_name']
        phone = request.POST['phone_number']

        myuser = User.objects.create_user(username, email, password1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.phone_number = phone

        myuser.save()

        messages.success(request, f'Account created for {username}!')

        return redirect('admin_account')

    return render(request, 'myapp/admin_account.html')


def remove_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        try:
            u = User.objects.get(username=username, is_superuser=False)
            u.delete()
            messages.success(
                request, f"The user {u.first_name} {u.last_name} deleted successfully!")
            return redirect('admin_account')
        except User.DoesNotExist:
            return redirect('admin_account')
        except Exception as e:
            return redirect('admin_account')
    return redirect('admin_account')


def submit_complaint(request):
    if request.method == 'POST':
        # Get form data
        footage = request.FILES.get('footage')
        comments = request.POST['comments']
        username = request.user.username

        # Create new complaint instance and save to database
        new_complaint = Complaints(
            footage=footage, comments=comments, username=username)
        new_complaint.save()

        messages.success(request, 'Complaint submitted successfully!')
        return redirect('user_account')

    return render(request, 'myapp/user_account.html')


def update_model(request):

    # replace with your instance retrieval logic
    mymodel_instance = CNN_Models.objects.first()
    if mymodel_instance is None:
        mymodel_instance = CNN_Models(eye_model=None, mouth_model=None)
    if request.method == 'POST':
        eyesmodel_file = request.FILES.get('eyesmodel')
        mouthmodel_file = request.FILES.get('mouthmodel')

        if eyesmodel_file:
            mymodel_instance.eye_model.save(
                eyesmodel_file.name, eyesmodel_file)
            messages.success(request, "Eyesmodel updates successfully!")

        if mouthmodel_file:
            mymodel_instance.mouth_model.save(
                mouthmodel_file.name, mouthmodel_file)
            messages.success(request, "Mouth updates successfully!")

        mymodel_instance.save()

        return redirect('admin_account')

    return render(request, 'myapp/admin_account.html')


def load_complaints(request):
    complaints = Complaints.objects.all().order_by('-date_created')
    data = []
    for complaint in complaints:
        data.append({
            'username': complaint.username,
            'comments': complaint.comments,
            'footage': complaints.footage,
            'date_created': complaint.date_created.strftime('%Y-%m-%d %H:%M:%S')
        })
    return JsonResponse(data, safe=False)


def get_models(request):
    models = CNN_Models.objects.first()
    data = []
    for model in models:
        data.append({
            'name': 'MODEL01',
            'mouth_model': model.mouth_model,
            'eye_model': model.eye_model,
        })
    return JsonResponse(data, safe=False)


def load_users(request):
    users = User.objects.all()
    user_list = []
    for user in users:
        user_list.append({
            'id': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': user.email,
        })
    return JsonResponse(user_list, safe=False)

# helpers----------------------->


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            if user.is_superuser:
                # return render(request,'myapp/admin_panel.html',{"username":username})
                return redirect('admin_account')
            else:
                return redirect('user_account')
        else:
            error = 'Invalid login credentials. Please try again.'
            return render(request, 'myapp/home.html', {'error': error})
    return render(request, 'myapp/home.html')
