import json, os, random,time,re
import numpy as np
from pathlib import Path

#default yaml file and ll temporal folder are put under the folder 'projects'
proj_all_path = 'projects/'
kirparser_path = '/home/arboratorgrew/autogramm/parsers/BertForDeprel/'

#the name of conda environment for each parser
kirian_env = 'kirian'
hops_env = 'base'

# os.system('conda init')
#todo custom yaml for hopsparser
#todo check conllu format
#should I remove the {info} in the name of train/dev/pred file?

def prepare_folder(project_name, train_set, conll_to_pred, dev_set = None):
    # suppose project_name doesn't end with '/', checked in fct train_pred
    # make temporal project folder
    # info = project_name[:-1] if project_name[-1] == '/' else info
    conll_path =  proj_all_path + project_name +'/' +'conllus/'
    Path( conll_path ).mkdir(parents=True, exist_ok=True)

    # make dataset
    dev_set = 0.2 if dev_set is None else dev_set
    if isinstance(dev_set, float):
        all_data = np.array(train_set.strip().split('\n\n')) #all conllu sentence in training set
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
    with open( conll_path +f'to_pred_{project_name}.conllu' , 'w') as f:
        print('end of to_pred', conll_to_pred[-2:] )
        #conll_to_pred = conll_to_pred if conll_to_pred[-2:] == '\n' else conll_to_pred+'\n'
        f.write( conll_to_pred.strip()+'\n\n' ) 

    return conll_path[:-8] #project path
           


def train_pred_hops(project_path, info):
    #make res folder
    Path( project_path+'hops_res/predicted/' ).mkdir(parents=True, exist_ok=True)
    conll_path =  project_path+'conllus/'

    #switch to the relevant environment
    #os.system(f"conda activate {hops_env}") #how to correctly init env with django?

    #train
    os.system(f'/home/arboratorgrew/miniconda3/bin/hopsparser train {proj_all_path}/example.yaml {conll_path}train_{info}.conllu \
            {project_path}hops_res/ \
            --dev-file \"{conll_path}dev_{info}.conllu\" \
            --device \"cuda:0\"')
            
    # #predict   
    parsed_path = f'{project_path}hops_res/predicted/parsed_{info}.conllu'
    os.system(f'/home/arboratorgrew/miniconda3/bin/hopsparser parse {project_path}hops_res/model/ \
            {conll_path}to_pred_{info}.conllu  {parsed_path}')
    return parsed_path


def copy_pos(origin_path, pred_path):
    print('coping original UPOS')
    origin = [t.split('\n') for t in open(origin_path).read().strip().split('\n\n') if t]
    pred = [t.split('\n') for t in open(pred_path).read().strip().split('\n\n') if t]

    origin_dict = {}
    pred_dict = {}

    for conllu in origin:
        key = conllu[0].split('=')[1].strip()
        origin_dict[key] = conllu[1:]

    for conllu in pred:
        key = conllu[0].split('=')[1].strip()
        pred_dict[key] = conllu[1:]

    for key, conll in pred_dict.items():
        for l, line in enumerate(conll):
            if(line[0]!='#'):
                info = line.split('\t')
                info_tag = origin_dict[key][l].split('\t')
                #print(info)
                info[3] = info_tag[3]
                pred_dict[key][l] = '\t'.join(info)
        pred_dict[key] = '\n'.join(pred_dict[key])
    
    to_write ='\n\n'.join([f'# sent_id = {key}\n'+val for k, val in pred_dict.items()]) + '\n\n'
    with open(pred_path, 'w') as f:
        f.write(to_write)

    return to_write




def train_pred_kirian(project_path, info):
    #make res folder
    # Path( project_path+'hops_res/predicted/' ).mkdir(parents=True, exist_ok=True)

    #switch to the relevant environment
    #os.system(f"conda activate {kirian_env}")

    # Create the annotation schema
    os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3 {kirparser_path}BertForDeprel/preprocessing/1_compute_annotation_schema.py \
        --input_folder \"{project_path}conllus\" \
        --output_path \"{project_path}annotation_schema.json\"")
        
    # Train  
    os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3 {kirparser_path}BertForDeprel/run.py train \
        --folder \"{project_path}\" \
        --ftrain \"{project_path}conllus/train_{info}.conllu\" \
        --ftest \"{project_path}conllus/dev_{info}.conllu\" \
        --model \"kirparser_{info}.pt\" \
        --batch 16 \
        --gpu_ids 0 \
        --epochs 60 \
        --punct \
        --bert_type  \"bert-base-multilingual-uncased\"")


    #Predict 
    os.system(f"/home/arboratorgrew/miniconda3/envs/kirian/bin/python3  {kirparser_path}BertForDeprel/run.py predict \
        --folder \"{project_path}\" \
        --fpred \"{project_path}conllus/to_pred_{info}.conllu\" \
        --multiple \
        --model  \"kirparser_{info}.pt\" \
        --batch 16 \
        --gpu_ids 0 \
        --punct")

    return project_path+f'predicted/to_pred_{info}.conllu'
    

    

#todo: an option keep to keep specific column in conllu (currently copy past the origine value to the predicted one)
def train_pred( project_name, train_set, conll_to_pred, dev_set = None, parser = 'hops', keep_pos = True):
    """
    project_name: string
    train_set, conll_to_pred, dev_set: a sequence of conllu seperated by '\n\n' end with '' or '\n'
    if dev_set == None, split train set, if dev_set == ratio, split train set with this ratio
    parser: type of parser

    @return: the prediction of conll_to_pred
    """
    print('parser = ',parser)

    #make temporal project folder
    info = project_name[:-1] if project_name[-1] == '/' else project_name
    project_path = prepare_folder(info, train_set, conll_to_pred, dev_set)
    conll_path = project_path+'conllus/'

    tmp = time.time()
    if(parser.lower() in ['hops', 'loic']):  
        parsed_path = train_pred_hops(project_path, info)
    
    elif(parser.lower() in ['kirian', 'bertfordeprel', 'kir']):
        parsed_path = train_pred_kirian(project_path, info)

    else:
        return 'Error: unknown parser type'

    print(f'\ntrain and prediction done, taken time {time.time() - tmp}s')
    print(parsed_path)   
    #for current version return all   
    #origin_path, pred_path
    return copy_pos(f'{project_path}conllus/to_pred_{info}.conllu', parsed_path) if keep_pos else open(parsed_path).read() 
<<<<<<< HEAD

=======
>>>>>>> 53edaef16d810171a0becf0e4dbbc968f39f7d49

