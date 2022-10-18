# pip install trankit
# https://trankit.readthedocs.io/en/stable/training.html

import trankit,re, os, sys, json

#load pipe
from trankit import Pipeline
from trankit.utils import CoNLL
from trankit.iterators.tagger_iterators import TaggerDataset
from trankit.utils.base_utils import get_ud_score, get_ud_performance_table

from processData import  make_data_dict, replace_col, dict2conll
from manage_parse import logging, remove_project
from pathlib import Path

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

TO_PARSE_NAMES = 'to_parse_fnames.tsv'
EVAL_DEV_NAME = 'evalDev' 


def has_mwt(train_path, mwt_batch_size = 50):
    conll_list = CoNLL.load_conll(open(train_path), ignore_gapping = True)

    count = 0
    for sent in conll_list:
        for l in sent:
            if '-' in l[0]:
                count += 1
            if count > mwt_batch_size:
                return True
    return False


def train_deprel_feat(res_folder, train_path, dev_path, epoch, category = 'customized-mwt' ):
    # initialize a trainer for the task
    trainer_dep = trankit.TPipeline(
        training_config={
        'max_epoch': epoch,
        'category': category, # pipeline category
        'task': 'posdep', # task name
        'save_dir': res_folder, # directory for saving trained model
        'train_conllu_fpath': train_path, # annotations file in CONLLU format  for training
        'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
        'embedding': 'xlm-roberta-large'

        }
    )

    # start training
    trainer_dep.train()
    return trainer_dep



def train_lemma(res_folder, train_path, dev_path, epoch, category = 'customized-mwt'):
    # initialize a trainer for the task
    trainer= trankit.TPipeline(
        training_config={
            'max_epoch': epoch,
            'category': category,  # pipeline category
            'task': 'lemmatize', # task name
            'save_dir': res_folder, # directory for saving trained model
            'train_conllu_fpath': train_path, # annotations file in CONLLU format  for training
            'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
            'embedding': 'xlm-roberta-large'
        }
    )
    # start training
    trainer.train()


def get_raw_file(conllu_path, raw_path):
    txt = open(conllu_path).read()
    txt_pattern = re.compile(r"# text =.+")
    res = '\n'.join([l[9:] for l in re.findall(txt_pattern, txt)])
    if raw_path:
        with open(raw_path, 'w') as f:
            f.write(res)


def train_tok(res_folder, train_path, dev_path, train_raw, dev_raw, epoch_tok, category = 'customized-mwt' ):
    get_raw_file(train_path, train_raw)
    get_raw_file(dev_path, dev_raw)
    
    # initialize a trainer for the task
    trainer_tok = trankit.TPipeline(
        training_config={
            'max_epoch': epoch_tok,
            'category': category, # pipeline category
            'task': 'tokenize', # task name
            'save_dir': res_folder, # directory for saving trained model
            'train_txt_fpath': train_raw, # raw text file
            'train_conllu_fpath': train_path, # annotations file in CONLLU format for training
            'dev_txt_fpath': dev_raw, # raw text file
            'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
            'embedding': 'xlm-roberta-large'
        }
    )
    # start training
    trainer_tok.train()

def train_mwt(res_folder,  train_path, dev_path, epoch_tok):    
    # initialize a trainer for the task
    trainer_mwt = trankit.TPipeline(
        training_config={
            'max_epoch': epoch_tok,
            'category': 'customized-mwt', # pipeline category
            'task': 'mwt', # task name
            'save_dir': res_folder, # directory for saving trained model
            'train_conllu_fpath': train_path, # annotations file in CONLLU format for training
            'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
            'embedding': 'xlm-roberta-large'
        }
    )
    # start training
    trainer_mwt.train()



def check_pipe(model_dir, category):
    #check pipe
    trankit.verify_customized_pipeline(
        category= category, # pipeline category
        save_dir= model_dir, # directory used for saving models in previous steps
        embedding_name='xlm-roberta-large' # embedding version that we use for training our customized pipeline, by default, it is `xlm-roberta-base`
    )


def train_trankit(res_folder, train_path, dev_path, train_raw, dev_raw, epoch, epoch_tok, category):
    #tokenize
    train_tok(res_folder, train_path, dev_path, train_raw, dev_raw, epoch_tok, category)
    if 'mwt' in category:
        train_mwt(res_folder, train_path, dev_path, epoch_tok)
    #posdep
    train_deprel_feat(res_folder, train_path, dev_path, epoch, category)
    #lemma
    train_lemma(res_folder,train_path, dev_path, min(20, epoch), category) #set max epoch as 20
    
    check_pipe(res_folder, category)


def get_tok_ls(conllu_path):
    conll_list = CoNLL.load_conll(open(conllu_path), ignore_gapping = True)

    tok_ls = []
    expand_end = -1 
    for sent in conll_list:
        sent_info = []
        ldict = {}
        for l in sent:
            if '-' in l[0]:
                ldict['id'] = ( int(l[0].split('-')[0]), int(l[0].split('-')[1]) )
                ldict['text'] = l[1]
                expand_end = ldict['id'][1]
                ldict['expanded'] = []
            elif expand_end > 0 and int(l[0]) <= expand_end:
                ldict['expanded'].append( { 'id': int(l[0]), 'text' : l[1] } )

                if int(l[0]) == expand_end:
                    #reset
                    expand_end = -1
                    sent_info.append(ldict)
                    ldict = {}
            else:
                sent_info.append( { 'id': int(l[0]), 'text' : l[1] }) 
        tok_ls.append(sent_info)
    return tok_ls


def pred_trankit( pipe , to_parse_path, parsed_path, task = 'tokenize'):

    if task == 'tokenize':
        #if task == tokenize, input file is txt file
        to_tok_txt = open(to_parse_path).read()
        res_dict = pipe.tokenize(to_tok_txt)

        doc_conll = CoNLL.convert_dict([ s['tokens'] for s in res_dict['sentences'] ], use_expand = True)
        conll_string = CoNLL.conll_as_string(doc_conll)
        print("PRED TOKENIZED\n", conll_string[:500])
        with open(parsed_path, 'w') as outfile:
            outfile.write(conll_string)
        return
    
    #lemmatize + pos headid tag feat
    tok_ls = get_tok_ls(to_parse_path)

    if task == 'posdep':
        res_dict = pipe.posdep_withID(tok_ls )

        doc_conll = CoNLL.convert_dict([ s['tokens'] for s in res_dict['sentences'] ], use_expand = True)
        conll_string = CoNLL.conll_as_string(doc_conll)
        print("PRED POSDEP\n ", conll_string[:500])
        with open(parsed_path, 'w') as outfile:
            outfile.write(conll_string)

    if task == 'lemmatize':
        res_dict = pipe.lemmatize_withID(tok_ls)

        doc_conll = CoNLL.convert_dict([ s['tokens'] for s in res_dict['sentences'] ], use_expand = True)
        conll_string = CoNLL.conll_as_string(doc_conll)
        print("PRED LEMMA\n",conll_string[:500] )
        with open(parsed_path, 'w') as outfile:
            outfile.write(conll_string)


def eval_parsed(parsed_path, gold_path):
    score = get_ud_score(parsed_path, gold_path)
    print(get_ud_performance_table(score))
    return score

def save_score(score,score_dir,  res_folder, cv_idx, name = 'test', newfile = True):
    metric_ls = [ "Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS",
                   "CLAS", "MLAS", "BLEX"]
    mode = 'w' if newfile else 'a'
    #f1 score
    with open(os.path.join(score_dir, name+'_trankit_f1score.tsv'), mode) as f:
        if mode == 'w':
            f.write('\t'.join( ["Metrics"] +  metric_ls) + '\n')
        
        f.write('\t'.join( [f"cv{cv_idx}"] + ["{}".format(score[metric].f1) for metric in metric_ls] ) + "\n" )

    #more score
    res = {}
    for metric in metric_ls:
        res[metric] = {
            'precision': score[metric].precision,
            'recall': score[metric].recall,
            'f1': score[metric].f1}
        if score[metric].aligned_accuracy is not None:
            res['aligned_accuracy'] = score[metric].aligned_accuracy

    with open(os.path.join(res_folder, name + '_score.json'), 'w') as f1:
        json.dump(res, f1, indent = 4)


# may not be useful,  only for posdep task
def test_deprel(trainer, test_path, name = 'test_dep'):
    #test trainer 
    #from trankit.iterators.tagger_iterators import TaggerDataset
    #trainer should be TPipeline instance for posdep, not for lemma 

    test_set = TaggerDataset(
        config=trainer._config,
        input_conllu = test_path,
        gold_conllu= test_path,
        evaluate= False
    )

    test_set.numberize()
    test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
    result = trainer._eval_posdep(data_set=test_set, batch_num=test_batch_num,
                            name=name, epoch= -1)
    print(trankit.utils.base_utils.get_ud_performance_table(result[0]))
    return result[0]

def posdep(res_folder, epoch, test_path, name = 'test_dep'):
    print(name)
    trainer = train_deprel_feat(res_folder, epoch)
    score = test_deprel(trainer, test_path, name = name)
    return score


def pred_trankit_toked(pipe, to_parse_path, res_folder):
    for task in ['posdep','lemmatize']:
        print('==== pred for task ', task, " ", to_parse_path)
        parsed_path = os.path.join( res_folder, f'{task}_{os.path.basename(to_parse_path)}')
        pred_trankit(pipe, to_parse_path, parsed_path, task = task)


def pred_trankit_raw(pipe, to_parse_path, res_folder):
    #tokenize:
    print("deug0 res_fd: ",res_folder)
    fname = ".".join(os.path.basename(to_parse_path).split('.')[:-1])
    tokenized_path = os.path.join( res_folder, f'tokenized_{fname}.conllu')
    print("debug1: tok_path", tokenized_path)
    if to_parse_path[-7:] == '.conllu':
        #input is conllu format, extract raw text from # text = ...
        txt_path = os.path.join( os.path.dirname(res_folder), f'{fname}_to_tok.txt')
        print("debug2: txt_path",txt_path)
        get_raw_file(to_parse_path, txt_path)
        pred_trankit(pipe, txt_path, tokenized_path, task = "tokenize")
    else:
        #is txt input
        pred_trankit(pipe, to_parse_path, tokenized_path, task = "tokenize")

    for task in ['posdep','lemmatize']:
            print('==== pred for task ', task)
            parsed_path = os.path.join(res_folder, f'{task}_{fname}.conllu')
            print(parsed_path)
            pred_trankit(pipe, tokenized_path, parsed_path, task = task)


def pred_trankit_toked_list(res_folder, to_parse_list, to_parse_dir, category = 'customized' ):
    #res_folder: e.g. trankit_res, folder to store custom training res
    p = Pipeline(lang= category, cache_dir= res_folder, embedding = 'xlm-roberta-large' )

    for fname in to_parse_list:
        #fname with .conllu format
        pred_trankit_toked(p, os.path.join(to_parse_dir,fname), res_folder)


def pred_trankit_raw_list(res_folder, to_parse_list, to_parse_dir, category = 'customized' ):
    #res_folder: e.g. trankit_res, folder to store custom training res
    p = Pipeline(lang= category, cache_dir= res_folder, embedding = 'xlm-roberta-large' )

    for fname in to_parse_list:
        #fname with .conllu format
        pred_trankit_raw(p, os.path.join(to_parse_dir,fname), res_folder)



def trankit_outfile(to_parse_dir, fname , res_folder, epochs , tokenized = True):
    # combine res to parsed_path, return parsed_path
    if tokenized:
        conllu_str = open( os.path.join( to_parse_dir, fname)).read().strip()
        dict_out = make_data_dict(conllu_str, uid_to_add = f'trankitParser{epochs}' )

        posdep_str = open( os.path.join( res_folder, f'posdep_{fname}')).read().strip()
        dict_posdep = make_data_dict(posdep_str, uid_to_add = f'trankitParser{epochs}' )
        dict_out = replace_col(dict_posdep, dict_out, [UPOS, XPOS, FEATS, HEAD, DEPREL])

        lemma_str = open( os.path.join( res_folder, f'lemmatize_{fname}')).read().strip()
        dict_lemma = make_data_dict(lemma_str, uid_to_add = f'trankitParser{epochs}' )
        dict_out = replace_col(dict_lemma, dict_out, [LEMMA])
    else:
        # parsed from raw
        # !!! won't be applied for arboratorgew, because tokenization task require linguistic knowledge. 
        conllu_str = open( os.path.join( res_folder, f'posdep_{fname}')).read().strip()
        dict_out = make_data_dict(conllu_str, uid_to_add = f'trankitTokParser{epochs}' )

        lemma_str = open( os.path.join( res_folder, f'lemmatize_{fname}')).read().strip()
        dict_lemma = make_data_dict(lemma_str, uid_to_add = f'trankitTokParser{epochs}' )
        dict_out = replace_col(dict_lemma, dict_out, [LEMMA])

        conllu_com = open( os.path.join( to_parse_dir, fname)).read().strip()
        comment_dict = make_data_dict(conllu_com, uid_to_add = f'trankitTokParser{epochs}' )
        # print("\n\n====DEBUG comment_dict\n")
        for idx, sent in dict_out.items():
            if idx not in comment_dict.keys():
                print("BUG idx", idx)
                print(sent)
                continue
            if '# sent_id' in '\n'.join(comment_dict[idx]['comment']):
                dict_out[idx]['comment'] = comment_dict[idx]['comment'] 

    return dict2conll(dict_out)



def train_pred_trankit(project_path, parser_id,  epochs = 5, epochs_tok = 5, need_train = True, tokenized = True, parse_train = True):
    print("trankit train pred")
    res_folder = os.path.join( project_path, f"{parser_id}_res")
    Path( res_folder).mkdir(parents=True, exist_ok=True)

    train_path = os.path.join(res_folder, 'conllus/train.conllu') 
    dev_path = os.path.join(res_folder, 'conllus/dev.conllu') 
    dev_raw = os.path.join(res_folder, 'conllus/dev_raw.txt') 
    train_raw = os.path.join(res_folder, 'conllus/train_raw.txt') 

    category = 'customized'
    if not tokenized:
        mwt =  has_mwt(dev_path)
        if mwt:
            mwt = has_mwt(train_path)
        category = 'customized-mwt' if mwt else 'customized'

    # Train  
    if need_train:
        if tokenized:
            logging( project_path, f'Training trankit model, category = {category}, epochs = {epochs}\n') 
        else:
            logging( project_path, f'Training trankit model, category = {category}, epochs = {epochs}, type = trankitTokParser\n') 
        train_trankit(res_folder, train_path, dev_path, train_raw, dev_raw, epochs, epochs_tok, category) 
    #Parse
    logging( project_path, f'Parse files\n')
    input_path = os.path.join(project_path, 'input_conllus')
    to_pred_path = os.path.join( input_path, 'to_parse') 

    to_parse_names = open( os.path.join( project_path, TO_PARSE_NAMES)).read().strip().split('\t')
    # to_parse_list = [ f for f in os.listdir(to_pred_path) if f[-7:] == '.conllu' ] 
    to_parse_list = [  f + '.conllu' for f in to_parse_names if f ]
    train_list = [f for f in os.listdir(input_path) if f[-7:] == '.conllu']

    predicted_path = os.path.join( res_folder,  "predicted/") 
    Path( predicted_path).mkdir(parents=True, exist_ok=True)
    #TODO check if we can put raw text file in arboratorgrew

    eval_path = os.path.join( res_folder, EVAL_DEV_NAME )
    Path( eval_path ).mkdir(parents=True, exist_ok=True)

    p = Pipeline(lang= category, cache_dir= res_folder, embedding = 'xlm-roberta-large' )

    if tokenized:
        #parse files to parse
        for fname in to_parse_list:
            #fname with .conllu format
            pred_trankit_toked(p, os.path.join(to_pred_path,fname), res_folder)

            conll_out = trankit_outfile(to_pred_path, fname , res_folder, epochs)
            with open( os.path.join(predicted_path, fname) , 'w') as outf:
                outf.write(conll_out)

        if parse_train:
            for fname in train_list:
                pred_trankit_toked(p, os.path.join(input_path,fname), res_folder)

                conll_out = trankit_outfile(input_path, fname , res_folder, epochs)
                with open( os.path.join(predicted_path, fname) , 'w') as outf:
                    outf.write(conll_out)
        #eval
        pred_trankit_toked(p, dev_path, res_folder)
        conll_out = trankit_outfile( os.path.join(res_folder, 'conllus'), 'dev.conllu' , res_folder, epochs)
        with open( os.path.join(eval_path, 'dev.conllu') , 'w') as outf:
            outf.write(conll_out)
    else:
        for fname in to_parse_list:
            #fname with .conllu format
            pred_trankit_raw(p, os.path.join(to_pred_path,fname), res_folder)

            conll_out = trankit_outfile(to_pred_path, fname , res_folder, epochs, tokenized=False)
            with open( os.path.join(predicted_path, fname) , 'w') as outf:
                outf.write(conll_out)
        #parse train file
        # pred_trankit_raw_list(res_folder, train_list, to_parse_dir = input_path, category = category  )
        if parse_train:
            for fname in train_list:
                pred_trankit_raw(p, os.path.join(input_path,fname), res_folder)

                conll_out = trankit_outfile(input_path, fname , res_folder, epochs, tokenized=False)
                with open( os.path.join(predicted_path, fname) , 'w') as outf:
                    outf.write(conll_out)
        #eval
        pred_trankit_raw(p, dev_path, res_folder)
        conll_out = trankit_outfile( os.path.join(res_folder, 'conllus'), 'dev.conllu' , res_folder, epochs, tokenized=False)
        with open( os.path.join(eval_path, 'dev.conllu') , 'w') as outf:
            outf.write(conll_out)

    return predicted_path




if __name__ == '__main__':
    # pred_fpath = 'trankit/pred_test_lem.conllu'
    if len(sys.argv) < 8:
        print(len(sys.argv))
        print("Usage: train_pred_trankit.py project_path parser_id need_train epochs epochs_tok tokenized parse_train", file=sys.stderr)
        sys.exit(-1)

    #set param
    project_path = sys.argv[1]
    parser_id = sys.argv[2]
    res_folder = os.path.join(sys.argv[1], f'{parser_id}_res')
    
    need_train = True if sys.argv[3].lower() == 'true' else False
    epochs = int(sys.argv[4])
    epochs_tok = int(sys.argv[5])
    tokenized = True if sys.argv[6].lower() == 'true' else False
    parse_train = True if sys.argv[7].lower() == 'true' else False

    print(type(need_train),type(tokenized), type(parse_train))
    print(need_train, tokenized, parse_train)

    # print(epochs, epochs_tok, type(epochs), type(epochs_tok))
    predicted_path = train_pred_trankit(
        project_path, parser_id,
        epochs = epochs,
        epochs_tok = epochs_tok,
        need_train = need_train,
        tokenized = tokenized,
        parse_train = parse_train
        )

   
