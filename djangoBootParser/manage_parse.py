### functions to manage training etc.###
import json, os, random,time,re, zipfile
import numpy as np
from pathlib import Path


#default yaml file and ll temporal folder are put under the folder 'projects'
PROJ_ALL_PATH = 'projects/'
TO_PARSE_NAMES = 'to_parse_fnames.tsv'
EVAL_DEV_NAME = 'evalDev'  # eval folder name, under {parser_id}_res folder
# score_path = os.path.join(eval_path, f'{parser_id}_f1score.json')

#log path and parsed path to modified according to project 
LOG_NAME = 'progress.txt'
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)
# PARSED_PATH = ''


def logging(project_path, log_info, begin = False):
    if begin:
        Path( project_path ).mkdir(parents=True, exist_ok=True)
        with open( os.path.join(project_path, LOG_NAME), 'w') as mylog:
            print('logbegin: ', log_info)
            mylog.write(log_info)
    else:
        with open( os.path.join(project_path, LOG_NAME), 'a') as mylog:
            print('add: ', log_info)
            mylog.write(log_info)
    if 'error' in log_info.lower():
        remove_project(os.path.basename(project_path))

def get_progress(project_fdname, parser_id):
    project_path = os.path.join( PROJ_ALL_PATH, project_fdname )
    try:
        prog = open( os.path.join(project_path, LOG_NAME) ).read().strip().split('\n')
        print('current log=====')
        print(os.path.join(project_path, LOG_NAME) )

        res = ''
        if(prog[-1].lower() == "fin"):
            res = _get_results(project_path, parser_id)
        # os.system(f'remove {LOG_PATH + LOG_NAME}')
        return (prog[-1], res)
    except FileNotFoundError:
        remove_project(project_fdname)


    

def _get_results(project_path, parser_id ):
    conll_files = []
    conllu_list = open( os.path.join( project_path, TO_PARSE_NAMES)).read().strip().split('\t')
    conllu_list += [f[:-7] for f  in os.listdir(os.path.join( project_path, 'input_conllus')) if f[-7:] == '.conllu' ] #parsed train set
    # conllu_list =  os.listdir(PARSED_PATH)
    
    print(conllu_list)
    parsed_path = os.path.join( project_path, f'{parser_id}_res/predicted' )
    for filename in conllu_list:
        conll_files.append(open(  os.path.join(parsed_path, filename + '.conllu') ).read())  
    score_dev = json.load( open( os.path.join(project_path, f'{parser_id}_res' , EVAL_DEV_NAME, f'{parser_id}_f1score.json') ) )
    return (conllu_list, conll_files, score_dev )

def remove_project(project_fdname):
    print(f'Remove {project_fdname}')
    os.system(f'rm -r { os.path.join( PROJ_ALL_PATH , project_fdname) }')
