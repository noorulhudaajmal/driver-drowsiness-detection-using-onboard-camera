from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
                  path("", views.home, name="home"),
                  path('userpanel/', views.user_panel, name='user_panel'),
                  path('login/', views.user_login, name='login'),
                  path('logout/', views.user_logout, name='logout'),
                  path('register/', views.register_user, name='register'),
                  path('submit-complaint/', views.submit_complaint, name='submit_complaint'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
