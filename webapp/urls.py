from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("", views.home, name="home"),
    path("home/", views.home, name="home"),
    path('admin_account/', views.admin_account, name='admin_account'),
    path('user_account/', views.user_account, name='user_account'),
    path('register_user/', views.register_user, name='register_user'),
    path('remove_user/', views.remove_user, name='remove_user'),
    path('submit_complaint/', views.submit_complaint, name='submit_complaint'),
    path('update_model/', views.update_model, name='update_model'),
    path('login_view/', views.login_view, name='login_view'),


    # path("login/", views.log_in, name="log_in"),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
