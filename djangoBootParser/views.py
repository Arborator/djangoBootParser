from django.http import HttpResponse
from rest_framework.decorators import api_view
# from django.shortcuts import render
# from django.contrib.auth.models import User, Group
# from typometricsapp.serializers import UserSerializer, GroupSerializer
from rest_framework import viewsets, status
from rest_framework.response import Response
from djangoBootParser.parse import train_pred

 
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.Try to add conllus in urls.")


#only POST is accepted
@api_view(['POST'])
def conllus(request):
    if request.method == 'POST':
        #get param
        proj_name = request.data.get('project_name' , 'project_0')
        train_set = request.data.get('train_set' , None)
        to_parse = request.data.get('to_parse' , None)
        dev_set = request.data.get('dev' , None)
        parser_type = request.data.get('parser', 'hops')

        #check param
        if train_set is None:
            return Response({'Error':'empty train set error'}, status=status.HTTP_400_BAD_REQUEST)
        if to_parse is None:
            return Response({'Error':'empty file to parse'}, status=status.HTTP_400_BAD_REQUEST)

        #train and pred, receive parsed file if parser_type is valid 
        parsed_conllu = train_pred( proj_name, train_set, to_parse, dev_set, parser_type)

        if parsed_conllu[:5] == 'Error':
            return Response({'Error' : parsed_conllu},  status=status.HTTP_400_BAD_REQUEST )

        #reply with conllu file 
        return Response({'parsed_file' : parsed_conllu})
    else:
        return Response({'error':'Only accept POST request'}, status=status.HTTP_400_BAD_REQUEST)