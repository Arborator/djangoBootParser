
### I. prepare dataset ###
from genericpath import isdir
import json, zipfile, os, sys, time,random,re
import numpy as np
from pathlib import Path
import base64, hashlib

from yaml import parse # sha512
from djangoBootParser.manage_parse import logging, remove_project
import pandas as pd
from datetime import datetime


#global var for djangoBootParser
PROJ_ALL_PATH = 'projects/'
#log path and parsed path to modified according to project 
LOG_NAME = 'progress.txt'
TO_PARSE_NAMES = 'to_parse_fnames.tsv'
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)
sid_pattern = re.compile(r'# sent_id =(.+)')
uid_pattern = re.compile(r'# user_id =(.+)')
comment_pattern = re.compile( r'# .+' )
comment_pattern_tosub = re.compile( r'# .+\n' )

ERR_PATH = 'format_err.txt' #under project_path


def estimate_time(parser_id, len_dataset, epochs):
    param_time = pd.read_csv('estimated_time_100ep.tsv', sep = '\t', index_col = 0)

    if parser_id == 'trankitTokParser':
        #TODO time test for tokenization
        y = param_time['trankitParser']['a'] * len_dataset + param_time['trankitParser']['b']
        return int(epochs * y /100) + 2
    
    y = param_time[parser_id]['a'] * len_dataset + param_time[parser_id]['b']
    if parser_id == 'udifyParser':
        return int(epochs * (y - 20 )/100 ) + 20 + 2
    return int(epochs * y /100) + 2

def check_empty_file(fname_ls, conll_ls):
    if fname_ls is None or conll_ls is None:
        return None, None

    assert(len(fname_ls) == len(conll_ls))
    for idx in range(len(conll_ls)):
        if not conll_ls[idx]:
            print('Empty file: ', fname_ls[idx])
            fname_ls.remove(fname_ls[idx])
    conll_ls = [l for l in conll_ls if l]
    assert(len(fname_ls) == len(conll_ls))
    return fname_ls, conll_ls


def remove_old_project(capacity = 25):
    print('Capacity in nb of folders: ', capacity)
    fd_list = [fd for fd in os.listdir(PROJ_ALL_PATH) if fd not in ['example.yaml', LOG_NAME] ]

    if len(fd_list) > capacity:
        print("Remove old project to make more espace")
        #'timestamp.txt' wrote at the end of prepare_folder()
        timestamp_list = [( fd_path,  float(open(os.path.join( PROJ_ALL_PATH, fd_path, 'timestamp.txt' )).read().strip()) )  for fd_path in  fd_list ]
        # print(timestamp_list)
        time_list = sorted(timestamp_list, key = lambda x : datetime.fromtimestamp(x[1]))
        print(time_list)
        for rm_info in time_list[: -capacity]:
            os.system(f"rm -r { os.path.join( PROJ_ALL_PATH, rm_info[0]) }")


def sha512_foldername(conll_names, conll_set_list, parser_id, dev_set, epochs):
    input_string_ls = [str(parser_id), str(dev_set), str(epochs)]+conll_names
    for conll in conll_set_list:
        input_string_ls.append(re.sub(comment_pattern_tosub ,'', conll).strip()  )
    input_string = '\n\n'.join(input_string_ls)
    # base64.b64encode(hashlib.sha512(whateverString.encode()).digest())
    return hashlib.sha256(input_string.encode()).hexdigest()

def _project_exist(project_fdname, to_parse_info, parser_id):
    #project_path and to_parse_info produced with sha512
    res_path = os.path.join(PROJ_ALL_PATH, project_fdname)
    is_trained = os.path.exists( res_path )
    is_parsed = False
    # print(res_path)

    if is_trained:
        print(' The same  input file for training set: check file to parsed, parse unparsed files')          
        parsed_path = os.path.join(res_path, f'{parser_id}_res/predicted')
        is_parsed = os.path.exists( os.path.join(parsed_path , to_parse_info + '.txt') ) 
    return is_trained, is_parsed


def log_err(err_path, message):
    with open(err_path,'a') as f:
        f.write(message)

        
def check_format_conllu(conllu, err_path, parser_id, is_to_parse = False ):
    info = re.sub(comment_pattern_tosub ,'', conllu).strip().split('\n')
    correct = True

    for l in info:
        tok = l.split('\t')
        if len(tok) != 10:
            log_err(err_path, f"\n\nError: Expect 10 columns but there is {len(tok)} in {l} \r\n")
            correct = False
    if is_to_parse:
        return correct

    conllu_info = [l.split('\t') for l in info]
    
    root = [word_info for word_info in conllu_info if '-' not in word_info[ID] and int(word_info[HEAD]) == 0 ]
    if len(root) != 1:
        log_err(err_path, f"\n\nError: multi root with {root}\r\n")
        correct = False
    
    for i, linfo in enumerate(conllu_info):
        if '-' not in linfo[ID] and linfo[HEAD] == '_' :
            log_err(err_path, "\n\nError: headID with '_' at " + '\t'.join(linfo) + '\r\n')
            correct = False
        #   linfo[HEAD] = 1
    if not correct:
        log_err(err_path, f"checked for conllu begin with {info[0]} \n=====")
    return correct

def _check_format_hops_parse(conllu, err_path):
    info = re.sub(comment_pattern_tosub ,'', conllu).strip().split('\n')

    conllu_info = [l.split('\t') for l in info]
    head = [word_info[HEAD] for word_info in conllu_info if '-' not in word_info[ID]]
    if np.all(head == '_'):
        return conllu
    root= [int(h) for h in head if h != '_' and int(h) == 0]
    if len(root) !=1:
        for idx in range(len(conllu_info)):
            log_err(err_path, f"\n\nWronging: no single root with {root} in file to parse, replacing head by _ for hopsparser\r\n")
            conllu_info[idx][HEAD] = '_'
        conllu = '\n'.join(['\t'.join(l) for l in conllu_info])
        return conllu
    return conllu


def check_format(fname, conllu_string, err_path, parser_id,is_to_parse = False):
    conllu_sents = conllu_string.strip().split('\n\n')
    print('check_format')
    for conll in conllu_sents:
        if parser_id in ['trankitTokParser'] and is_to_parse:
            # if we parse from raw data and we check the files to parse
            print('check_format TOK')
            if '# text =' not in conll:
                log_err(f"In {fname}\nNo text to parse in {conll}")
        else:
            # if we check the gold files or if we will not do tokenization task 
            if not check_format_conllu(conll, err_path, parser_id, is_to_parse = is_to_parse):
                print("UDError in ", fname, " current conll length ", len(conllu_sents))
                # log_err(f"checking {fname}\n")
                conllu_sents.remove(conll)
    if parser_id == 'hopsParser':
        conllu_sents = [_check_format_hops_parse(c, err_path) for c in conllu_sents ]

    return '\n\n'.join(conllu_sents)



def _create_folder( project_path, parser_id, train_names, train_set_list, parse_names, to_parse_list, dev_set = 0.1):
    """ create training set 
    # project_path : created with sha512
    # parser_id = 'auto' or parser_id in ['hopsParser', 'kirParser', 'udifyParser', 'trankitParser', 'stanzaParser']
    # in auto mode, we choose parser between udify and trankit according to training set length. Current implementation can't 
    make dataset for stanza in auto mode, adapt it further if necessary.
    """
    print('\n\ncreate folder ')
    is_stanza = 'stanza' in parser_id
    res_path = os.path.join(project_path, f'{parser_id}_res')
    conll_path = os.path.join( res_path, 'conllus') if not is_stanza else os.path.join( res_path, 'extern_data/conllus/xxx_stanza/')
    Path( conll_path ).mkdir(parents=True, exist_ok=True)

    input_path = os.path.join(project_path, 'input_conllus')
    to_pred_path = os.path.join( input_path, 'to_parse') 
    Path( to_pred_path ).mkdir(parents=True, exist_ok=True)
    
    print(type(train_names), len(train_set_list), type(parse_names), len(to_parse_list))
    assert(len(train_names) == len(train_set_list) and len(parse_names) == len(to_parse_list))

    #store input
    err_path = os.path.join( project_path, ERR_PATH)
    # with open(err_path, 'w') as fbegin:
    #   fbegin.write(f'CHECK FORMAT BEGIN\n')
    print(err_path)
    train_set = []
    for filename, conll in zip(train_names, train_set_list):
        conll = check_format(filename, conll, err_path, parser_id)
        train_set.append(conll)
        with open( os.path.join( input_path,  filename + '.conllu'), 'w') as f:
            f.write(conll.strip() + '\n\n')

    for filename, conll in zip(parse_names, to_parse_list):
        conll = check_format(filename, conll, err_path, parser_id,is_to_parse= True)
        with open( os.path.join( to_pred_path, filename + '.conllu'), 'w') as f:
            f.write(conll.strip() + '\n\n')
    
    with open( os.path.join(project_path, TO_PARSE_NAMES), 'w' ) as f:
        f.write('\t'.join(parse_names))
    
    train_set = '\n\n'.join([ conll.strip() for conll in train_set])
    del train_set_list
    del to_parse_list
    
    # make dataset
    #all conllu sentence in training set, pay attention to the nombre of '\n' at the end of files
    # remove np.array(train_set.split('\n\n')) cost too much if train_set is large
    all_data = train_set.split('\n\n') 
    print(len(all_data))
    #sample
    idx_dev = random.sample(range(len(all_data)), k = int(len(all_data)*dev_set )) 
    #split
    idx_train = list(set(np.arange(len(all_data))) - set(idx_dev))
    dev_set_ls = [all_data[d].strip() for d in idx_dev]
    train_set_ls = [all_data[t].strip() for t in idx_train] 
    dev_set = '\n\n'.join(dev_set_ls) + '\n\n'
    train_set = '\n\n'.join(train_set_ls) + '\n\n'
    print(len(all_data))

    # store dataset
    train_path = os.path.join( conll_path, 'xxx_stanza-ud-train.conllu') if is_stanza else os.path.join( conll_path, f'train.conllu') 
    dev_path = os.path.join( conll_path, 'xxx_stanza-ud-dev.conllu') if is_stanza else os.path.join( conll_path, f'dev.conllu') 
    
    with open(  train_path, 'w') as f: 
        f.write( train_set )
    with open(  dev_path, 'w') as f: 
        f.write( dev_set ) 
    if 'stanza' in parser_id:
        with open( os.path.join(conll_path, 'xxx_stanza-ud-test.conllu') , 'w') as f: 
            f.write( dev_set )
    if parser_id == 'auto':
        parser_id = 'udifyParser' if len(all_data) < 100 else 'trankitParser'
        os.system(f"mv {res_path} {os.path.join(project_path, f'{parser_id}_res')}")
    logging( project_path, f'Parser ID = {parser_id}\n', begin = False)
    return parser_id, len(all_data)


def prepare_folder(project_name, train_names, train_set_list, parse_names, to_parse_list, parser_id = 'hopsParser', dev_set = 0.1, epochs = 100):
    """
    make temporal project folder
    by default we parse also the training set  

    parser_id = 'auto' or parser_id in ['hopsParser', 'kirParser', 'udifyParser', 'trankitParser', 'stanzaParser']

    return need_train, need_parse
    """

    print("Check path")
    #project name with sha512
    project_fdname = sha512_foldername(train_names, train_set_list, parser_id, dev_set, epochs) # train_set with config 
    to_parse_info = sha512_foldername(parse_names,to_parse_list, parser_id, dev_set, epochs)

    print(project_fdname, ' epochs:', epochs)
    is_trained, is_parsed = _project_exist(project_fdname, to_parse_info, parser_id)
    # print(is_trained, is_parsed)
    if is_trained and is_parsed:
        print('already trained and parsed')
        return False, False, project_fdname, to_parse_info, parser_id, -1

    try:
        # Define path
        project_path = os.path.join(PROJ_ALL_PATH, project_fdname)
        res_path = os.path.join(project_path, f'{parser_id}_res')
        Path( res_path ).mkdir(parents=True, exist_ok=True)

        to_pred_path = os.path.join( project_path, 'input_conllus', 'to_parse') 
        Path( to_pred_path ).mkdir(parents=True, exist_ok=True)

        print('check done')
        if is_trained and not is_parsed:
            logging( project_path, ': Already trained for current input, storing file to parse\n', begin = True)

            #store files to parse
            for filename, conll in zip(parse_names, to_parse_list):
                with open( os.path.join( to_pred_path,  filename + '.conllu'), 'w') as f:
                    f.write(conll.strip() + '\n\n')
            #store names of files to parse 
            with open( os.path.join(project_path, TO_PARSE_NAMES), 'w' ) as f:
                f.write('\t'.join(parse_names))
            if parser_id == 'auto':
                os.system(f"rm -r {res_path}")
                fdname = [ fd for fd in  os.listdir(project_path) if '_res' in fd  ]
                print(fdname )
                parser_id = fdname[0].split('_')[0]
            return False, True, project_fdname, to_parse_info, parser_id, -1

        # create training set 
        logging( project_path, 'Preparing dataset\n', begin = True)
        parser_id, len_dataset= _create_folder( project_path, parser_id, train_names, train_set_list, parse_names, to_parse_list, dev_set = dev_set)
        # TODO time test for tokenization task
        estim_time = estimate_time(parser_id, len_dataset, epochs)
        logging( project_path, f'Estimated time: {estim_time} min\n') 
        # add or update timestamp
        with open(os.path.join( project_path, 'timestamp.txt'), 'w') as tf:
            now = datetime.now()
            print(now)
            tf.write( str(datetime.timestamp(now)) )
        remove_old_project(capacity = 25)
        return not is_trained, not is_parsed, project_fdname, to_parse_info, parser_id, estim_time   # == True, True
    except:
        logging( project_path, 'Error for data preparation\n') 
        remove_project(project_fdname)

