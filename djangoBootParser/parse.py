import json, os, random,time,re, zipfile, datetime
import numpy as np
from pathlib import Path
# sha512
import base64, hashlib

from djangoBootParser.manage_parse import logging, remove_project
from djangoBootParser.conll2018_eval import evaluate, load_conllu, load_conllu_file
from djangoBootParser.processData import copy_pos_file, add_uid_file, add_uid, add_multitok, make_data_dict, replace_col, dict2conll

# from djangoBootParser.train_pred_stanza import  train_stanza, pred_stanza_raw, pred_onlytok_stanza, pred_stanza_toked, pred_stanza_tagged, pretrain_wv
# from djangoBootParser.train_pred_trankit import train_trankit, pred_trankit_toked_list, pred_trankit_raw_list, has_mwt


#default yaml file and ll temporal folder are put under the folder 'projects'
PROJ_ALL_PATH = 'projects/'
TO_PARSE_NAMES = 'to_parse_fnames.tsv' # put it also in train_pred_stanza.py & train_pred_trankit.py
EVAL_DEV_NAME = 'evalDev'  # under {parser_id}_res folder

kirparser_path = '/home/arboratorgrew/autogramm/parsers/BertForDeprel/'
udify_path = '/home/arboratorgrew/autogramm/parsers/udify/'
stanza_abs_root =  '/home/arboratorgrew/autogramm/djangoBootParser/projects/'

#the name of conda environment for each parser
kirian_env = 'kirian'
hops_env = 'base'
udify_env = 'udify'

#TODO test stanza_tok and trankit_tok, integrate them into frontend
parserID_dict = {
    'kirian': 'kirParser',
    'hops': 'hopsParser',
    'udify' : 'udifyParser',
    'stanza': 'stanzaParser',
    'stanza_tok': 'stanzaTokParser',
    'trankit' :'trankitParser',
    'trankit_tok' : 'trankitTokParser'
}

#log path and parsed path to modified according to project 
LOG_NAME = 'progress.txt'
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

           
def train_hops(project_path, conll_path,  epochs = 5):
    #train
    logging( project_path, f'hopsParser: training model ({epochs} epochs in total)\n')
    print(PROJ_ALL_PATH)
    #TODO modify the source code of hops to set custom epochs and remove this part 
    config_expath = os.path.join(PROJ_ALL_PATH,  'example.yaml')
    if not os.path.exists(config_expath):
        config_expath = 'example.yaml'
        
    config = [l for l in open( config_expath).read().strip().split('\n') if l] #remove ''
    config[-1] = f'epochs: {epochs}'
    with open( os.path.join( project_path,  'config.yaml'), 'w') as f:
        f.write('\n'.join(config))

    res_path = os.path.join(project_path, f"{parserID_dict['hops']}_res")

    os.system(f"/home/arboratorgrew/miniconda3/bin/hopsparser train {os.path.join(project_path, 'config.yaml')} \
            {os.path.join( conll_path, 'train.conllu')} \
            {res_path} \
            --dev-file \"{os.path.join( conll_path, 'dev.conllu')}\" \
            --device \"cuda:0\"")
    return


def train_pred_hops(project_path, epochs = 5, need_train = True, parse_train = True):
    #make res folder
    res_path = os.path.join(project_path, f"{parserID_dict['hops']}_res")
    conll_path = os.path.join( res_path, 'conllus') 
    parsed_path = os.path.join( res_path, 'predicted/')
    Path( parsed_path ).mkdir(parents=True, exist_ok=True)

    # train
    if need_train:
        train_hops(project_path, conll_path, epochs = epochs)
            
    #predict   
    logging( project_path, f'hopsParser: model trained, parsing files ...\n') 

    input_path = os.path.join(project_path, 'input_conllus')
    to_pred_path = os.path.join( input_path, 'to_parse') 
    to_parse_names = open( os.path.join( project_path, TO_PARSE_NAMES)).read().strip().split('\t')
    # to_parse_list = [ f for f in os.listdir(to_pred_path) if f[-7:] == '.conllu' ] 
    to_parse_list = [ f + '.conllu' for f in to_parse_names if f] 

    for conll_to_pred in to_parse_list:
        print('parsing ', conll_to_pred)
        os.system(f"/home/arboratorgrew/miniconda3/bin/hopsparser parse {os.path.join(res_path, 'model')} \
            {os.path.join(to_pred_path, conll_to_pred)}  {os.path.join(parsed_path, conll_to_pred)}")

    if parse_train:
        train_list = [f for f in os.listdir(input_path) if f[-7:] == '.conllu']
        for conll_to_pred in train_list:
            print('parsing ', conll_to_pred)
            os.system(f"/home/arboratorgrew/miniconda3/bin/hopsparser parse {os.path.join(res_path, 'model')} \
                {os.path.join(input_path, conll_to_pred)}  {os.path.join(parsed_path, conll_to_pred) }")
    #eval on dev
    eval_path = os.path.join( res_path, EVAL_DEV_NAME)
    Path( eval_path ).mkdir(parents=True, exist_ok=True)
    os.system(f"/home/arboratorgrew/miniconda3/bin/hopsparser parse {os.path.join(res_path, 'model')} \
            {os.path.join( conll_path, 'dev.conllu')}  {os.path.join(eval_path, 'dev.conllu') }")

    return parsed_path


def train_kirian(project_path, epochs = 5):
    # Create the annotation schema
    res_path = os.path.join(project_path, f"{parserID_dict['kirian']}_res")
    os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3 {kirparser_path}BertForDeprel/preprocessing/1_compute_annotation_schema.py \
        --input_folder \"{os.path.join( res_path, 'conllus')}\" \
        --output_path \"{os.path.join( res_path, 'annotation_schema.json')}\"")
        
    # Train  
    logging( project_path, f'kirParser: training model ({epochs} epochs in total)\n')

    os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3 {kirparser_path}BertForDeprel/run.py train \
        --folder \"{res_path}\" \
        --ftrain \"{os.path.join( res_path, 'conllus/train.conllu')}\" \
        --ftest \"{os.path.join( res_path, 'conllus/dev.conllu')}\" \
        --model \"kirparser.pt\" \
        --batch 16 \
        --gpu_ids 0 \
        --epochs {epochs} \
        --punct \
        --bert_type  \"bert-base-multilingual-uncased\"") 
    return


def train_pred_kirian(project_path, epochs = 5, need_train = True, parse_train = True):
    #make res folder
    res_path = os.path.join(project_path, f"{parserID_dict['kirian']}_res")
    conll_path = os.path.join( res_path, 'conllus') 
        
    # Train  
    if need_train:
        train_kirian(project_path,  epochs = epochs)

    #Predict 
    logging( project_path, f'kirParser: model trained, parsing files ...\n')

    input_path = os.path.join(project_path, 'input_conllus')
    to_pred_path = os.path.join( input_path, 'to_parse') 

    to_parse_names = open( os.path.join( project_path, TO_PARSE_NAMES)).read().strip().split('\t')
    # to_parse_list = [ f for f in os.listdir(to_pred_path) if f[-7:] == '.conllu' ] 
    to_parse_list = [ f + '.conllu' for f in to_parse_names if f ] 

    for conll_to_pred in to_parse_list :
        print(conll_to_pred)
        to_parse_path = os.path.join(to_pred_path, conll_to_pred)
        parsed_path = os.path.join( res_path, 'predicted', conll_to_pred )
        os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3  {kirparser_path}BertForDeprel/run.py predict \
            --folder \"{res_path}\" \
            --fpred \"{to_parse_path}\" \
            --multiple \
            --overwrite  \
            --model  \"kirparser.pt\" \
            --batch 8 \
            --gpu_ids 0 \
            --punct")
        add_multitok(parsed_path, to_parse_path)

    if parse_train:
        train_list = [f for f in os.listdir(input_path) if f[-7:] == '.conllu']
        for conll_to_pred in train_list :
            print(conll_to_pred)
            to_parse_path = os.path.join(input_path, conll_to_pred)
            parsed_path = os.path.join( res_path, 'predicted' ,conll_to_pred )
            os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3  {kirparser_path}BertForDeprel/run.py predict \
                --folder \"{res_path}\" \
                --fpred \"{to_parse_path}\" \
                --multiple \
                --overwrite \
                --model  \"kirparser.pt\" \
                --batch 8 \
                --gpu_ids 0 \
                --punct")
            add_multitok(parsed_path, to_parse_path)
    
    #eval on dev
    eval_path = os.path.join( res_path, EVAL_DEV_NAME)
    Path( eval_path ).mkdir(parents=True, exist_ok=True)
    to_parse_path = os.path.join(conll_path, 'dev.conllu')
    parsed_path = os.path.join(eval_path, 'dev.conllu')
    os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3  {kirparser_path}BertForDeprel/run.py predict \
        --folder \"{res_path}\" \
        --fpred \"{to_parse_path}\" \
        --multiple \
        --overwrite \
        --model  \"kirparser.pt\" \
        --batch 8 \
        --gpu_ids 0 \
        --punct")
    os.system(f"mv {os.path.join(res_path, 'predicted', 'dev.conllu')}  {parsed_path}")
    add_multitok(parsed_path, to_parse_path)

    return os.path.join(res_path, 'predicted/')

#udify
def train_udify(project_path, epochs = 5):
    print("udify train")
    #define path
    res_path = os.path.join(project_path, f"{parserID_dict['udify']}_res")

    logging( project_path, f'udifyParser: training model ({epochs} epochs in total)\n')
    #train
    os.system(f"/home/arboratorgrew/miniconda3/envs/udify/bin/python3 {os.path.join(udify_path, 'train_autogram.py')} \
        --config config/test_udify.json \
        --base_config config/udify_base.json \
        --project_path {res_path}\
        --epochs {epochs} \
        --dataset_dir {os.path.join( res_path, 'conllus')}")
    return
    

def train_pred_udify(project_path,  epochs = 5, need_train = True, parse_train = True):
    print("udify train pred")
    tmp = time.time()
    # Train  
    res_path = os.path.join(project_path, f"{parserID_dict['udify']}_res")

    predicted_path = os.path.join( res_path,  "predicted/") 
    Path( predicted_path).mkdir(parents=True, exist_ok=True)

    if need_train:
        train_udify(project_path,   epochs = epochs)

    #Parse
    logging( project_path, f'udifyParser: model trained, parsing files ...\n')
    print("Udify train: taken time: ", time.time() - tmp)

    input_path = os.path.join(project_path, 'input_conllus')
    to_pred_path = os.path.join( input_path, 'to_parse') 

    to_parse_names = open( os.path.join( project_path, TO_PARSE_NAMES)).read().strip().split('\t')
    # to_parse_list = [ f for f in os.listdir(to_pred_path) if f[-7:] == '.conllu' ] 
    to_parse_list = [ os.path.join( to_pred_path ,f + '.conllu') for f in to_parse_names if f]
    train_list = [ os.path.join(input_path, f) for f in os.listdir(input_path) if f[-7:] == '.conllu']

    #eval on dev
    eval_path = os.path.join( res_path, EVAL_DEV_NAME)
    Path( eval_path ).mkdir(parents=True, exist_ok=True)
    to_eval_path = os.path.join(res_path, 'conllus', 'dev.conllu')
    # parsed_fpath = os.path.join(eval_path, 'dev.conllu')

    fpath_all = to_parse_list + train_list + [to_eval_path] if parse_train else to_parse_list + [to_eval_path]

    to_parse_all = [open(fpath).read().strip() for fpath in fpath_all ]
    print(fpath_all)
    sign = '\n\n# '+ '#'*128 + '\n'
    tmp_parse_path = os.path.join(project_path, 'tmp_to_parse.conllu')
    tmp_parsed_path = os.path.join( predicted_path, 'tmp_parsed.conllu' )
    with open( tmp_parse_path, 'w') as fparse:
        fparse.write( sign.join(to_parse_all) + '\n\n' )

    logging( project_path, f'udifyParser: parsing combined file...\n')
    os.system(f"/home/arboratorgrew/miniconda3/envs/udify/bin/python3 {os.path.join(udify_path, 'predict.py')} \
        {os.path.join( res_path, 'logs/model.tar.gz')} \
        {tmp_parse_path} \
        {tmp_parsed_path }" )
    print("Udify parse combined file taken time: ", time.time() - tmp)

    # complete the original comments
    parsed_str = open( tmp_parsed_path ).read().strip()
    dict_out = make_data_dict(parsed_str, uid_to_add = f'udifyParser{epochs}')

    conllu_str = open( tmp_parse_path ).read().strip()
    dict_gold = make_data_dict(conllu_str, uid_to_add = f'udifyParser{epochs}')
    dict_out = replace_col(dict_gold, dict_out, [], repl_comment = True)

    conll_out = dict2conll(dict_out)
    with open( tmp_parsed_path , 'w') as outf:
        outf.write(conll_out)

    # split parsed files
    logging( project_path, f'udifyParser: split parsed combined file...\n')
    parsed_all_txt = open(tmp_parsed_path).read().strip().split(sign)
    eval_f = parsed_all_txt[-1]
    print("DEBUG_UDIFY", len(eval_f))
    # store ...
    with open(os.path.join(eval_path, 'dev.conllu'), 'w') as fe:
        fe.write(eval_f.strip() + '\n\n')

    for idx, fpath in enumerate(fpath_all[:-1]):
        fname = os.path.join( predicted_path, os.path.basename(fpath))
        with open(fname, 'w') as fp:
            fp.write( parsed_all_txt[idx].strip() +'\n\n' )
    
    return predicted_path


# stanza

def train_pred_stanza(project_path, epochs = 5, need_train = True, keep_pos = True, tokenized = True,  epochs_tok = 5, parse_train = True):
    pid = 'stanza' if tokenized else 'stanza_tok'

    os.system(f"/home/arboratorgrew/miniconda3/bin/python3 djangoBootParser/train_pred_stanza.py {project_path} {parserID_dict[pid]} \
        {need_train} {epochs} {epochs_tok} {keep_pos} {tokenized} {parse_train}")

    return os.path.join( project_path,  f"{parserID_dict[pid]}_res", "predicted/") 
    

def train_pred_trankit(project_path,  epochs = 5, epochs_tok = 5, need_train = True, tokenized = True, parse_train = True):
    pid = 'trankit' if tokenized else 'trankit_tok'
    os.system(f"/home/arboratorgrew/miniconda3/bin/python3 djangoBootParser/train_pred_trankit.py {project_path} {parserID_dict[pid]} \
        {need_train} {epochs} {epochs_tok}  {tokenized} {parse_train}")
    return os.path.join( project_path,  f"{parserID_dict[pid]}_res", "predicted/") 
    
def eval_parsed(parsed_path, gold_path):
    parsed = load_conllu_file( parsed_path )
    gold = load_conllu_file( gold_path )
    score = evaluate(gold, parsed)
    return score


def add_upos_uid(to_parse_fpath, parsed_path, user_id, keep_pos = True):
    # repl_comment = 'udify' in parser_id
    if keep_pos:
        print("Copy UPOS from input file to parsed results & add user_id = ", user_id)
        for conll_to_pred in to_parse_fpath:
            print('process ', conll_to_pred)
            conllu_str = open( conll_to_pred).read().strip()
            dict_gold = make_data_dict(conllu_str, uid_to_add = user_id)

            fname = os.path.join( parsed_path, os.path.basename(conll_to_pred))
            parsed_str = open( fname ).read().strip()
            dict_out = make_data_dict(parsed_str, uid_to_add = user_id )
            dict_out = replace_col(dict_gold, dict_out, [UPOS]) #, repl_comment = repl_comment)
            
            conll_out = dict2conll(dict_out)
            with open( fname, 'w') as outf:
                outf.write(conll_out)

    else:
        print("Add user_id = ", user_id)
        for conll_to_pred in to_parse_fpath:
            fname = os.path.join( parsed_path, os.path.basename(conll_to_pred))
            parsed_str = open( fname).read().strip()
            dict_out = make_data_dict(parsed_str, uid_to_add = user_id)

            # if repl_comment:
            #     conllu_str = open( conll_to_pred).read().strip()
            #     dict_gold = make_data_dict(conllu_str, uid_to_add = user_id)
            #     dict_out = replace_col(dict_gold, dict_out, [], repl_comment = repl_comment)

            conll_out = dict2conll(dict_out)
            with open( fname , 'w') as outf:
                outf.write(conll_out)

def score_on_dev(parsed_dev_path, dev_path, parser_id, eval_path):
    score = eval_parsed(parsed_dev_path, dev_path)

    metric_ls = ["UPOS", "UAS", "LAS"] if parser_id in ["hopsParser", "kirParser"] else [ "Tokens", "Sentences", "Words", "UPOS", "XPOS","UFeats", "Lemmas", "UAS", "LAS"]
    res = {}
    for metric in metric_ls:
        res[metric] = {
            'precision': score[metric].precision,
            'recall': score[metric].recall,
            'f1': score[metric].f1}

    score_path = os.path.join(eval_path, f'{parser_id}_f1score.json')  
    with open( score_path, 'w') as f1:
        json.dump(res, f1, indent = 4)
    return res

#todo: an option keep to keep specific column in conllu (currently copy past the origine value to the predicted one)
def train_pred( project_name,  project_fdname, to_parse_info, parser_id = 'hopsParser', keep_pos = True, epochs = 5, need_train = True, parse_train = True ):
    """
    project_name: string
    train_set, to_pred, dev_set: a sequence of conllu seperated by '\n\n' end with '' or '\n'
    if dev_set == ratio, split train set with this ratio
    parser: type of parser

    @return: the prediction of to_parse
    """
    print('parser = ',parser_id)
    tmp = time.time()
    try:
        #temporal project folder, keep the same path as the fct prepare_folder!
        project_path = os.path.join( PROJ_ALL_PATH , project_fdname )

        is_stanza = 'stanza' in parser_id
        res_path = os.path.join(project_path, f'{parser_id}_res')
        conll_path = os.path.join( res_path, 'conllus') if not is_stanza else os.path.join( res_path, 'extern_data/conllus/xxx_stanza/')

        input_path = os.path.join(project_path, 'input_conllus')
        to_pred_path = os.path.join( input_path, 'to_parse') 


        logging( project_path, 'Dataset prepared\n')

        #train & pred
        tmp = time.time()
        # global PARSED_PATH
        if parser_id == 'hopsParser':  
            parsed_path  = train_pred_hops(project_path, epochs = epochs, need_train = need_train, parse_train = parse_train)
        
        elif parser_id == 'kirParser':
            parsed_path  = train_pred_kirian(project_path, epochs = epochs, need_train = need_train,  parse_train = parse_train)

        elif parser_id == 'stanzaParser':
            parsed_path  = train_pred_stanza(project_path, epochs = epochs, need_train = need_train, keep_pos =keep_pos, parse_train = parse_train )
        
        elif parser_id == 'trankitParser':
            parsed_path  = train_pred_trankit(project_path,  epochs = epochs, need_train = need_train, parse_train = parse_train)

        elif parser_id == 'udifyParser':
            parsed_path  = train_pred_udify(project_path,  epochs = epochs, need_train = need_train, parse_train = parse_train)
        elif parser_id == 'trankitTokParser':
            parsed_path  = train_pred_trankit(project_path,  epochs = epochs, need_train = need_train, tokenized = False, parse_train = parse_train)
        else:
            return 'Error: unknown parser type'
        
        #note that we already parsed this file list
        with open(os.path.join(parsed_path , to_parse_info + '.txt'), 'w') as f:
            f.write(str(datetime.datetime.now())) 

        print(f'\ntrain and prediction done, taken time {time.time() - tmp}s')

        print("PARSED_PATH: ", parsed_path)   
        #for current version return all  

        logging( project_path, 'Files parsed, adding user_id etc.\n')
        
        input_path = os.path.join(project_path, 'input_conllus')
        to_pred_path = os.path.join( input_path, 'to_parse') 

        to_parse_names = open( os.path.join( project_path, TO_PARSE_NAMES)).read().strip().split('\t')
        # to_parse_list = [ os.path.join(to_pred_path, f) for f in os.listdir(to_pred_path) if f[-7:] == '.conllu' ] 
        to_parse_list = [ os.path.join(to_pred_path, f + '.conllu') for f in to_parse_names if f]
        train_list = [ os.path.join(input_path, f) for f in os.listdir(input_path) if f[-7:] == '.conllu']

        to_parse_fpath = to_parse_list + train_list
        print(to_parse_fpath)
        keep_upos = parser_id != 'trankitTokParser'
        add_upos_uid(to_parse_fpath = to_parse_fpath, parsed_path = parsed_path, user_id = parser_id+str(epochs), keep_pos = keep_upos)
        
        logging( project_path, 'user_id etc. added, evaluate on dev set \n')
        #evaluate
        eval_path = os.path.join( res_path, EVAL_DEV_NAME)
        dev_path = os.path.join( conll_path, 'xxx_stanza-ud-dev.conllu') if is_stanza else os.path.join( conll_path, 'dev.conllu' )
        parsed_dev_path = os.path.join(eval_path, 'xxx_stanza-ud-dev.conllu') if is_stanza else os.path.join( eval_path, 'dev.conllu' )
        score_dict = score_on_dev(parsed_dev_path, dev_path, parser_id, eval_path)
        print(score_dict)
        print(f"end for {parser_id}, taken time {time.time()-tmp} s")

        logging( project_path, 'Fin\n')    
        return parsed_path
    except:
        logging( project_path, 'Error in train or parse\n') 
        remove_project(project_fdname)
        
    
