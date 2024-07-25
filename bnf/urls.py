from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('chat/', chat, name='chat'),
    path('chat_conversation/', chat_conversation, name='chat_conversation'),
    #path('response/', response, name='response'),
    #path('chat_conversation/', chat_conversation, name='chat_conversation')

]