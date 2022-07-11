import json, os, random,time,re, zipfile
import numpy as np
from pathlib import Path

#default yaml file and ll temporal folder are put under the folder 'projects'
proj_all_path = 'projects/'
kirparser_path = '/home/arboratorgrew/autogramm/parsers/BertForDeprel/'
#project_path = proj_all_path + project_name + / 

#the name of conda environment for each parser
kirian_env = 'kirian'
hops_env = 'base'

#log path and parsed path to modified according to project 
LOG_NAME = 'progress.txt'
# PARSED_PATH = ''

#TODO
#todo custom yaml for hopsparser
#todo check conllu format
#should I remove the {info} in the name of train/dev/pred file?

def _logging(project_path, log_info, begin = False):
    if begin:
        Path( project_path ).mkdir(parents=True, exist_ok=True)
        with open( project_path + LOG_NAME, 'w') as mylog:
            print('logbegin: ', log_info)
            mylog.write(log_info)
    else:
        with open( project_path + LOG_NAME, 'a') as mylog:
            print('add: ', log_info)
            mylog.write(log_info)

def get_progress(project_name, parser_id):
    project_path = proj_all_path + project_name + '/'
    prog = open( project_path + LOG_NAME).read().strip().split('\n')
    print('current log=====')
    print(project_path + LOG_NAME)

    res = ''
    if(prog[-1].lower() == "fin"):
        res = _get_results(project_path, parser_id)
        # os.system(f'remove {LOG_PATH + LOG_NAME}')
    return (prog[-1], res)

def get_parsed_path(project_path, parser_id):
    if parser_id == 'hopsParser':  
        return  f'{project_path}hops_res/predicted/'    
    elif parser_id  'kirParser':
        return project_path+f'predicted/'


def _get_results(project_path, parser_id ):
    conll_files = []

    data_info =   open(project_path  + 'info.tsv').read().split('\n')
    conllu_list =[fname for fname in data_info[1].split('\t')[1:] if fname] #avoid ''
    # conllu_list =  os.listdir(PARSED_PATH)

    parsed_path = get_parsed_path(project_path, parser_id)
    for filename in conllu_list:
        conll_files.append(open(parsed_path + filename + '.conllu').read())  #check if .strip() + '\n'(or '\n\n'?? ) needed
    return (conllu_list, conll_files)

def remove_project(project_name):
    os.system(f'rm -r {proj_all_path + project_name}/')


def _project_exist(project_path, to_pred_path, train_names, train_set_list, parse_names, to_parse_list, parser_id):
    """
    case ex : condition description : futur action 
    case 1 : project_path not existe: create 
    case 2 : different input file for training set: backup and recreate
    case 3 : the same  input file for training set: check file to parsed, parse unparsed files  
    case 4 : different parser type: create 
    return : 
    True, []     if we need to make training set and train a model,
    False, [list of file to parse]    if no need
    """
    print('\n\ncheck project_path : ', project_path)
    if os.path.exists(project_path) and os.path.isfile(project_path+'info.tsv'):
        #check if training files or files to parse are the same
        data_info =   open(project_path + 'info.tsv').read().split('\n')
        old_train_names = data_info[0].split('\t')[1:]
        if parser_id != data_info[2].split('\t')[-1]:
            print('case 4 : different parser type: create') 
            return True, None

        #training set 
        conllu_list =  os.listdir( to_pred_path )
        print('old train:', old_train_names)
        existed = [] #[tname for tname in train_names if tname in old_train_names ]
        for idx, tname in enumerate(train_names):
            print(idx, tname)
            print(tname in old_train_names)
            if tname in old_train_names and open(to_pred_path + tname +'.conllu').read().strip() == train_set_list[idx].strip():
                print(tname)
                existed.append(tname)

        print(existed, train_names)
        if len(existed) != len(train_names):
            print('case 2 : different input file for training set: backup and recreate')
            print(project_path, '  ', project_path[:-1] + '_old/')
            os.system( f"mv {project_path} {project_path[:-1] + '_old/' }")
            return True, None

        else:
            print('case 3 : the same  input file for training set: check file to parsed, parse unparsed files') 
            to_parse = parse_names.copy()
            for idx, tname in enumerate(parse_names):
                if tname + '.conllu' in conllu_list and open(to_pred_path + tname +'.conllu').read().strip() == to_parse_list[idx].strip():
                    to_parse.remove(tname)
            return False, to_parse
        
    print('case 1 : project_path not existe: create')
    return True, None 

def _create_folder(to_pred_path, conll_path, project_name, train_names, train_set_list, parse_names, to_parse_list, dev_set = 0.2):
    # create training set 
    print('\n\ncreat folder ')
    Path( to_pred_path ).mkdir(parents=True, exist_ok=True)

    print(type(train_names), len(train_set_list), type(parse_names), len(to_parse_list))
    assert(len(train_names) == len(train_set_list) and len(parse_names) == len(to_parse_list))

    for filename, conll in zip(train_names, train_set_list):
        with open( to_pred_path + filename + '.conllu', 'w') as f:
            f.write(conll.strip() + '\n\n')

    for filename, conll in zip(parse_names, to_parse_list):
        with open( to_pred_path + filename + '.conllu', 'w') as f:
            f.write(conll.strip() + '\n\n')

    train_set = '\n\n'.join([ conll.strip() for conll in train_set_list])
    del train_set_list
    del to_parse_list
    
    # make dataset
    if isinstance(dev_set, float):
        #all conllu sentence in training set, pay attention to the nombre of '\n' at the end of files
        all_data = np.array(train_set.split('\n\n')) 
        #sample
        idx_dev = random.sample(range(len(all_data)), k = int(len(all_data)*dev_set )) 
        #split
        idx_train = list(set(np.arange(len(all_data))) - set(idx_dev))
        dev_set = '\n\n'.join(all_data[idx_dev]) + '\n\n'
        train_set = '\n\n'.join(all_data[idx_train]) + '\n\n'
        
    # store
    with open( conll_path +f'train_{project_name}.conllu' , 'w') as f: 
        f.write( train_set )
    with open( conll_path +f'dev_{project_name}.conllu' , 'w') as f: 
        f.write( dev_set ) 
    return


def prepare_folder(project_name, train_names, train_set_list, parse_names, to_parse_list, parser_id = 'hopsParser', dev_set = 0.2):
    """
    suppose project_name doesn't end with '/', checked in fct train_pred
    make temporal project folder
    by default we parse also the training set  
    """
    #info = project_name[:-1] if project_name[-1] == '/' else info
    project_path = proj_all_path + project_name + '/'
    conll_path =  project_path  +'conllus/'
    to_pred_path = conll_path + 'to_parse/' 

    print("Check path")
    need_train, just_parse_list = _project_exist(project_path, to_pred_path, train_names, train_set_list, parse_names, to_parse_list, parser_id)

    print('check done')
    if not need_train:
        _logging( project_path, 'Already trained for current input, storing file to parse\n', begin = True)

        #store unparsed files 
        for filename, conll in zip(parse_names, to_parse_list):
            if filename in just_parse_list :
                with open( to_pred_path + filename + '.conllu', 'w') as f:
                    f.write(conll.strip() + '\n\n')
        #update info about files to parse
        data_info =   open(project_path + 'info.tsv').read().split('\n')
        data_info[1] = ('to_parse\t' + '\t'.join(just_parse_list) +'\n')
        with open( project_path + 'info.tsv', 'w') as f:
            f.write('\n'.join(data_info))

        return False, just_parse_list

    # create training set 
    _logging( project_path, 'Preparing dataset\n', begin = True)
    _create_folder(to_pred_path, conll_path, project_name, train_names, train_set_list, parse_names, to_parse_list, dev_set)

    with open( project_path + 'info.tsv', 'w') as f:
        f.write('training\t' + '\t'.join(train_names) +'\n')
        f.write('to_parse\t' + '\t'.join(parse_names) +'\n')
        f.write(f'parser_id\t{parser_id}')

    return True, just_parse_list  # == True, None
           
def train_hops(project_path, conll_path, info):
    #train
    _logging( project_path, f'hopsParser: training model\n')#({epochs} epochs in total)\n')

    os.system(f'/home/arboratorgrew/miniconda3/bin/hopsparser train {proj_all_path}/example.yaml {conll_path}train_{info}.conllu \
            {project_path}hops_res/ \
            --dev-file \"{conll_path}dev_{info}.conllu\" \
            --device \"cuda:0\"')
    return


def train_pred_hops(project_path, info, just_parse_list = None):
    #make res folder
    Path( project_path+'hops_res/predicted/' ).mkdir(parents=True, exist_ok=True)
    conll_path =  project_path+'conllus/'
    to_pred_path = conll_path + 'to_parse/'

    # train
    if just_parse_list is None:
        train_hops(project_path, conll_path, info)
            
    #predict   
    _logging( project_path, f'hopsParser: model trained, parsing files ...\n') 

    parsed_path = f'{project_path}hops_res/predicted/'
    conllu_list =  [fname + '.conllu' for fname in just_parse_list] if just_parse_list is not None else os.listdir(to_pred_path) 
    # conllu_list.remove('SAY_BC_CONV_01.conllu') #remove training set 

    for conll_to_pred in conllu_list :
        print('parsing ', conll_to_pred)
        os.system(f'/home/arboratorgrew/miniconda3/bin/hopsparser parse {project_path}hops_res/model/ \
            {to_pred_path}{conll_to_pred}  {parsed_path}{conll_to_pred}')

    # os.system(f'/home/arboratorgrew/miniconda3/bin/hopsparser parse {project_path}hops_res/model/ \
    #         {conll_path}to_pred_{info}.conllu  {parsed_path}')
    return parsed_path


def train_kirian(project_path, info, epochs = 5):
    # Create the annotation schema
    os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3 {kirparser_path}BertForDeprel/preprocessing/1_compute_annotation_schema.py \
        --input_folder \"{project_path}conllus\" \
        --output_path \"{project_path}annotation_schema.json\"")
        
    # Train  
    _logging( project_path, f'kirParser: training model ({epochs} epochs in total)\n')

    os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3 {kirparser_path}BertForDeprel/run.py train \
        --folder \"{project_path}\" \
        --ftrain \"{project_path}conllus/train_{info}.conllu\" \
        --ftest \"{project_path}conllus/dev_{info}.conllu\" \
        --model \"kirparser_{info}.pt\" \
        --batch 16 \
        --gpu_ids 0 \
        --epochs {epochs} \
        --punct \
        --bert_type  \"bert-base-multilingual-uncased\"") 
    return


def train_pred_kirian(project_path, info, epochs = 5, just_parse_list = None):
    #make res folder
    # Path( project_path+'hops_res/predicted/' ).mkdir(parents=True, exist_ok=True)
        
    # Train  
    if just_parse_list is None:
        train_kirian(project_path, info, epochs = epochs)

    #Predict 
    _logging( project_path, f'kirParser: model trained, parsing files ...\n')

    to_pred_path = project_path+'conllus/to_parse/'
    conllu_list =  [fname + '.conllu' for fname in just_parse_list] if just_parse_list is not None else os.listdir(to_pred_path) 
    # conllu_list.remove('SAY_BC_CONV_01.conllu') #remove training set 

    for conll_to_pred in conllu_list :
        os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3  {kirparser_path}BertForDeprel/run.py predict \
            --folder \"{project_path}\" \
            --fpred \"{to_pred_path}{conll_to_pred}\" \
            --multiple \
            --model  \"kirparser_{info}.pt\" \
            --batch 16 \
            --gpu_ids 0 \
            --punct")

    return project_path+f'predicted/'
    


def copy_pos_file(origin_path, parsed_dict, filename):
    print('copy pos:',filename)
    origin_txt = open(origin_path + filename).read().strip()
    begin, tmp = origin_txt.split("sent_id ", 1)
    origin = [t.split('\n') for t in ("# sent_id "+tmp).split('\n\n') if t]

    origin_dict = {}

    for conllu in origin:
        # every sent begin with #sent_id 
        # TODO replace this by keyword sent_id instead of index 
        key = conllu[0].split('=')[1].strip()
        origin_dict[key] = [line for line in conllu[1:] if line[0] != '#']

    for key, conll in parsed_dict.items():
        begin = 0
        for l, line in enumerate(conll):
            if(line[0]!='#'):
                info = line.split('\t')
                info_tag = origin_dict[key][l - begin].split('\t')
                #print(info)
                info[3] = info_tag[3]
                parsed_dict[key][l] = '\t'.join(info)
            else:
                begin += 1 
        parsed_dict[key] = '\n'.join(parsed_dict[key])
    
    return parsed_dict

def add_uid_file(filename, parsed_path , parser_id, origin_path = None):
    print('--',filename)
    parsed_txt = open(parsed_path  + filename).read().strip()
    begin, tmp = parsed_txt.split("sent_id ", 1)
    parsed = [t.split('\n') for t in ("# sent_id "+tmp).split('\n\n') if t]

    parsed_dict = {}

    uid = f'# user_id = {parser_id}\n'

    for conllu in parsed:
        key = conllu[0].split('=')[1].strip()
        parsed_dict[key] = conllu[1:]

    if origin_path:
        parsed_dict = copy_pos_file(origin_path, parsed_dict, filename)

    to_write = begin[:-2] + '\n\n'.join([f'# sent_id = {k}\n'+ uid + val for k, val in parsed_dict.items()]) + '\n\n'
    with open(parsed_path  + filename, 'w') as f:
        f.write(to_write)

    return to_write

def add_uid(parser_id, parsed_path, origin_path = None, just_parse_list = None):
    #The last function to call after prediction & postprocessing
    #origin_path to keep original upos, should end with '/'
    conll_files = []
    conllu_list =  [fname + '.conllu' for fname in just_parse_list] if just_parse_list is not None else os.listdir(parsed_path)

    for filename in conllu_list:
        conll_files.append(add_uid_file(filename, parsed_path, parser_id, origin_path = origin_path))

    return (conllu_list, conll_files)



#todo: an option keep to keep specific column in conllu (currently copy past the origine value to the predicted one)
def train_pred( project_name,  info, parser_id = 'hopsParser', keep_pos = True, epochs = 5, just_parse_list = None ):
    """
    project_name: string
    train_set, to_pred, dev_set: a sequence of conllu seperated by '\n\n' end with '' or '\n'
    if dev_set == ratio, split train set with this ratio
    parser: type of parser

    @return: the prediction of to_parse
    """
    print('parser = ',parser_id)
    #temporal project folder, keep the same path as the fct prepare_folder!
    # info = project_name[:-1] if project_name[-1] == '/' else project_name
    # project_path = prepare_folder(info, train_name, train_set, parse_name, to_parse, dev_set)
    project_path = proj_all_path + project_name + '/'
    conll_path = project_path+'conllus/'

    _logging( project_path, 'Dataset prepared\n')

    #train & pred
    tmp = time.time()
    # global PARSED_PATH
    if parser_id == 'hopsParser':  
        parsed_path  = train_pred_hops(project_path, info, just_parse_list = just_parse_list)
    
    elif parser_id == 'kirParser':
        parsed_path  = train_pred_kirian(project_path, info, epochs = epochs, just_parse_list = just_parse_list)

    else:
        return 'Error: unknown parser type'

    print(f'\ntrain and prediction done, taken time {time.time() - tmp}s')

    print("PARSED_PATH: ", parsed_path)   
    #for current version return all  

    _logging( project_path, 'Files parsed, adding user_id etc.\n')
    if keep_pos:
        add_uid(parser_id, parsed_path, f'{conll_path}to_parse/', just_parse_list = just_parse_list)
    else:
        add_uid(parser_id, parsed_path, just_parse_list = just_parse_list)

    _logging( project_path, 'Fin\n')    
    return parsed_path
    
    



