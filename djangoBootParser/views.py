from django.http import HttpResponse
from rest_framework.decorators import api_view
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response
from djangoBootParser.parse import train_pred
from djangoBootParser.prepare_dataset import prepare_folder, check_empty_file
from djangoBootParser.manage_parse import get_progress,  remove_project
import json, threading, os


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
        keep_upos = request.data.get('keep_upos', False)

        #check param: remove potential empty file
        train_name, train_set = check_empty_file(train_name, train_set)
        parse_name, to_parse = check_empty_file(parse_name, to_parse)
        if train_set in [None, []]:
            return Response({'datasetStatus': 'Empty', 'error':'empty train set error'}, status=status.HTTP_400_BAD_REQUEST)
        if to_parse in [None, []]:
            return Response({ 'datasetStatus': 'Empty', 'error':'empty file to parse'}, status=status.HTTP_400_BAD_REQUEST)

        info = project_name[:-1] if project_name[-1] == '/' else project_name

        try:
            need_train, need_parse, project_fdname, to_parse_info, parser_ID, time = prepare_folder(info, train_name, train_set, parse_name, to_parse, parser_id, dev_set, epochs, keep_upos)
            print( 'dataset prepared, training is about to begin')
            if not need_train and not need_parse:
                return Response({'datasetStatus': 'OK', 'parseStatus': 'Done', 'time': -1 }) 
            if need_train and not need_parse:
                remove_project(project_fdname)
                return Response({'datasetStatus': '0', 'Error':'impossible case that is_trained = False but is_parsed = True '}, status=status.HTTP_400_BAD_REQUEST)
            # return Response({'datasetStatus': 'OK'})   
        except:
            return Response({'Error':'failed to prepare dataset'}, status=status.HTTP_400_BAD_REQUEST)

        #train and pred, receive parsed file if parser_id is valid 
        param = {
            'project_name': info,  
            'project_fdname' : project_fdname,
            'to_parse_info' : to_parse_info,
            'parser_id' : parser_ID, 
            'keep_upos': keep_upos,
            'epochs': epochs,
            'need_train' : need_train,
            'parse_train' : request.data.get('parse_train', False)
            }
        parser_thread = threading.Thread(target= train_pred, kwargs = param)
        parser_thread.start()

        err_info = None
        err_path = os.path.join( 'projects', project_fdname, 'format_err.txt' )
        if os.path.exists(err_path):
            err_info = open(err_path).read().strip()
        print('TIME ', time,'\n')
        return Response({'datasetStatus': 'OK', 'parseStatus': 'Begin', 'parserID':parser_ID,  'projectFolder': project_fdname, 'dataError': err_info, 'time': time}) 
    else:
        return Response({'error':'Only accept POST request'}, status=status.HTTP_400_BAD_REQUEST)



@api_view(['POST'])
def getResults(request):
    if request.method == 'POST':
        project_fdname = request.data.get('projectFdname', '')
        parser_id = request.data.get('parser', '')
        try:
            status, res = get_progress(project_fdname, parser_id)
        except:
            remove_project(project_fdname)
        if status.lower() == 'fin':
            parsed_names, parsed_conllu, devScore = res
            return Response({
                'status': status, 
                'logPath': project_fdname, 
                'parsed_names': parsed_names, 
                'parsed_files' : parsed_conllu,
                'devScore' : devScore,
                }) 
        else:
            return Response({'status': status, 'logPath': project_fdname }) 
    else:
        return Response({'error':'Only accept POST request'}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def removeParseFolder(request):
    if request.method == 'POST':
        project_fdname = request.data.get('projectFdname', '')
        remove_project(project_fdname)
        
        return Response({'status': 'OK'})  
    else:
        return Response({'error':'Only accept POST request'}, status=status.HTTP_400_BAD_REQUEST)

