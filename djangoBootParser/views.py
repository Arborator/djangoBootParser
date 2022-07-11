from django.http import HttpResponse
from rest_framework.decorators import api_view
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response
from djangoBootParser.parse import train_pred, prepare_folder, get_progress
import json, threading


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.Try to add conllus in urls.")

#for test
@api_view(['GET','POST'])
def testBoot(request):
    if request.method in ['GET', 'POST']:
        response = Response({"test":"hello"})
        return response
    else:
        return Response({'error':'connextion test failed'}, status=status.HTTP_400_BAD_REQUEST)

#only POST is accepted
@api_view(['POST'])
def conllus(request):
    if request.method == 'POST':
        #get param
        project_name = request.data.get('project_name' , 'project_0')
        train_name = request.data.get('train_name', '')
        train_set = request.data.get('train_set' , None)
        parse_name = request.data.get('parse_name', '')
        to_parse = request.data.get('to_parse' , None)
        dev_set = request.data.get('dev' , None)
        parser_id = request.data.get('parser', 'hopsParser')
        epochs = request.data.get('epochs', 5)
        keep_upos = request.data.get('keep_upos', True)

        #check param
        if train_set is None:
            return Response({'Error':'empty train set error'}, status=status.HTTP_400_BAD_REQUEST)
        if to_parse is None:
            return Response({'Error':'empty file to parse'}, status=status.HTTP_400_BAD_REQUEST)

        info = project_name[:-1] if project_name[-1] == '/' else project_name

        try:
            need_train, just_parse_list = prepare_folder(info, train_name, train_set, parse_name, to_parse, parser_id, dev_set)
            #dataset prepared, training is about to begin
            if not need_train and just_parse_list == []:
                Response({'datasetStatus': 'OK', 'parseStatus': 'Done'}) 
            # return Response({'datasetStatus': 'OK'})   
        except:
            return Response({'error':'failed to prepare dataset'}, status=status.HTTP_400_BAD_REQUEST)

        #train and pred, receive parsed file if parser_id is valid 
        param = {
            'project_name': project_name,  
            'info' : info, 
            'parser' : parser_id, 
            'keep_pos': keep_upos,
            'epochs': epochs,
            'just_parse_list' : just_parse_list
            }
        parser_thread = threading.Thread(target= train_pred, kwargs = param)
        parser_thread.start()

        return Response({'datasetStatus': 'OK', 'parseStatus': 'Begin'}) 
    else:
        return Response({'error':'Only accept POST request'}, status=status.HTTP_400_BAD_REQUEST)



@api_view(['POST'])
def getResults(request):
    if request.method == 'POST':
        project_name = request.data.get('project_name', '')
        parser_id = request.data.get('parser', '')
        status, res = get_progress(project_name, parser_id)

        if status.lower() == 'fin':
            parsed_names, parsed_conllu = res
            return Response({'status': status , 'logPath': project_name, 'parsed_names': parsed_names, 'parsed_files' : parsed_conllu}) 
        else:
            return Response({'status': status, 'logPath': project_name }) 
    else:
        return Response({'error':'Only accept POST request'}, status=status.HTTP_400_BAD_REQUEST)




