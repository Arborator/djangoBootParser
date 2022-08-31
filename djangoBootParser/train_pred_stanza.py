#if file to parse is raw text
import stanza, sys, os, re
from stanza.utils.conll import CoNLL
from stanza.models.common.pretrain import Pretrain

from gensim.models.fasttext import FastText
from pathlib import Path
from manage_parse import logging, remove_project


stanza_utils_path = '/home/arboratorgrew/autogramm/parsers/stanza/stanza/utils'
abs_root = '/home/arboratorgrew/autogramm/djangoBootParser/'
python_path = '/home/arboratorgrew/miniconda3/bin/python3'
TO_PARSE_NAMES = 'to_parse_fnames.tsv'
EVAL_DEV_NAME = 'evalDev' 

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)


def get_sentences(conllu_path_ls):
    text = []
    for conllu_path in conllu_path_ls:
        txt = open(conllu_path).read()
        txt_pattern = re.compile(r"# text =.+")
        text += [l[9:] for l in re.findall(txt_pattern, txt)]

    train_dir = os.path.dirname(conllu_path_ls[0])
    train_file = os.path.join(train_dir, 'train_emb.txt')

    with open(train_file, 'w') as f:
        f.write( '\n'.join(text) )

    return train_file

### Pretrain word vectors
def make_pretrain_wv(model, wv_dir, vector_size=100):
    txt_path = os.path.join(wv_dir, 'xxx.vectors.txt')
    # xz_path = os.path.join(wv_dir, 'xxx.vectors.xz')
    pt_path =  os.path.join(wv_dir, 'xxx_emb.pt')
    
    with open(txt_path, 'w') as f:
        f.write(f"{len(model.wv)} {vector_size}\n")
        
    to_write = '\n'.join([ k + ' ' + ' '.join( map(str, model.wv[k])) for k in model.wv.key_to_index])
    
    with open( txt_path  , 'a') as f:
        f.write(to_write)
    #make .pt file
    pt = Pretrain(pt_path, txt_path)
    pt.load()
    # os.system(f"rm txt_path")
    return pt_path
    

def pretrain_wv( train_path_ls, wv_dir = "data/wordvec/fasttext/Custom_Lang", vec_size = 100):
    print("Pretrain word vector...")
    Path( wv_dir).mkdir(parents=True, exist_ok=True)

    train_path_ls = [train_path_ls] if isinstance(train_path_ls, str) else train_path_ls
    train_file = get_sentences(train_path_ls)
    
    model = FastText(vector_size= vec_size, min_count = 1, epochs = 30)

    # build the vocabulary
    print("bulding vocab")
    model.build_vocab(corpus_file= train_file)

    # train the model
    print("train_model")
    model.train(
        corpus_file=train_file, epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words
    )
    #saved_models
    print("Storing pretrained word vector model to ", wv_dir)
    return make_pretrain_wv(model, wv_dir, vector_size= vec_size)


### Train
def train_tok(test_fd,  batch_size = 8,  eval_steps = 100, steps = 500):
    os.system(f"{python_path} {stanza_utils_path}/datasets/prepare_tokenizer_treebank.py xxx_stanza")
    os.system(f"{python_path} {stanza_utils_path}/training/run_tokenizer.py xxx_stanza --steps {steps} --eval_steps {eval_steps} --batch_size {batch_size}")

def train_mwt(test_fd,  batch_size = 8,  epochs = 100):
    os.system(f"{python_path} {stanza_utils_path}/datasets/prepare_mwt_treebank.py xxx_stanza")
    os.system(f"{python_path} {stanza_utils_path}/training/run_mwt.py xxx_stanza --num_epoch {epochs} --batch_size {batch_size}")


def train_lemma(test_fd, batch_size = 8, epochs = 100):
    # each example is an example of lemma instead of sentences, so by observation, put batch size as 200 instead of 8
    os.system(f"{python_path} {stanza_utils_path}/datasets/prepare_lemma_treebank.py xxx_stanza")
    os.system(f"{python_path} {stanza_utils_path}/training/run_lemma.py xxx_stanza --num_epoch {epochs} --batch_size {batch_size} ") 


def train_pos(test_fd, pretrain_path = None, batch_size = 8, steps = 500, epochs = 100, eval_interval = 100):
    pretrain = f"--wordvec_pretrain_file {pretrain_path}" if pretrain_path else "--no_pretrain"
    os.system(f"{python_path} {stanza_utils_path}/datasets/prepare_pos_treebank.py xxx_stanza")
    os.system(f"{python_path} {stanza_utils_path}/training/run_pos.py xxx_stanza --epochs {epochs} {pretrain} --batch_size {batch_size} --eval_interval {eval_interval} ")


def train_dep(test_fd, pretrain_path = None, batch_size = 8,  epochs = 100, eval_interval = 100, with_gold = True):
    pretrain = f"--wordvec_pretrain_file {pretrain_path}" if pretrain_path else "--no_pretrain"
    use_gold = "--gold" if with_gold else ""
    os.system("pwd")
    print("DEBUG DEP", pretrain_path)

    os.system(f"{python_path} {stanza_utils_path}/datasets/prepare_depparse_treebank.py xxx_stanza {use_gold}")
    os.system(f"{python_path} {stanza_utils_path}/training/run_depparse.py xxx_stanza --epochs {epochs} {pretrain} --batch_size {batch_size} --eval_interval {eval_interval}")




def train_stanza(test_fd, len_trainset, pretrain_path = None, with_gold = True, batch_size = 8,  batch_size_tok = 8, epochs = 100, epochs_tok = 100):
    steps_tok = int( len_trainset * epochs_tok / batch_size_tok + .5)
    eval_steps_tok = int(len_trainset / batch_size_tok + .5)
    logging( os.path.dirname(test_fd), f'Training stanza pipeline...\n')
    train_tok(test_fd, steps = steps_tok, eval_steps = eval_steps_tok, batch_size= batch_size_tok)
    # if has_mwt:
    train_mwt(test_fd, epochs = epochs, batch_size= batch_size)

    train_lemma(test_fd, epochs = epochs, batch_size= 100)
    train_pos(test_fd,pretrain_path, batch_size = 64, epochs = epochs, eval_interval = len_trainset)
    logging( os.path.dirname(test_fd), f'Training dependency parser in stanza pipeline...\n')
    train_dep(test_fd,pretrain_path, batch_size = 64, epochs = epochs, eval_interval = len_trainset, with_gold = with_gold)


### Parse ###
def make_to_parsef(to_parse_path):
    to_parse_txt = open(to_parse_path).read().strip().split('\n\n')
    sents = [sent.split('\n') for sent in to_parse_txt  ]

    for sid, sent in enumerate(sents):
        for tid, tok in enumerate(sent):
            if tok[0] == '#':
                continue
            sents[sid][tid] = sents[sid][tid].split('\t')
            for k in [LEMMA, XPOS, FEATS]:
                sents[sid][tid][k] ='_'
            sents[sid][tid] = '\t'.join(sents[sid][tid])
    return '\n\n'.join([ '\n'.join(sent) for sent in sents ]) + '\n\n'


## 1. Input with gold tokenization and one of the gold upos/xpos/ufeat/lemma ##
def parse_tagged(nlp, to_parse_path, parsed_path):
    to_parse_doc = CoNLL.conll2doc(input_file = to_parse_path )
    processed_doc = nlp(to_parse_doc)

    with open(parsed_path, 'w') as f:
        f.write( CoNLL.doc2conll_text(processed_doc) )

def pred_stanza_tagged(to_parse_list,  parsed_dir, model_dir, pretrain_path):
    # todo keep language names that don't in stanza_resources
    print(pretrain_path)
    to_parse_list = [to_parse_list] if isinstance(to_parse_list, str) else to_parse_list
    # some language such as zh may have _ everywhere in lemma column
    lemma_path = os.path.join(model_dir,'saved_models/lemma/xxx_stanza_lemmatizer.pt')
    
    if os.path.exists(lemma_path):
        nlp_dep = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,pos,lemma,depparse', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            
            lemma_model_path = lemma_path,
            depparse_model_path= os.path.join(model_dir, 'saved_models/depparse/xxx_stanza_parser.pt'),
            depparse_pretrain_path = pretrain_path,
            tokenize_pretokenized=True,
            pretagged=True
            )
    else:
        nlp_dep = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,pos,lemma,depparse', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            
            depparse_model_path= os.path.join(model_dir, 'saved_models/depparse/xxx_stanza_parser.pt'),
            depparse_pretrain_path = pretrain_path,
            tokenize_pretokenized=True,
            pretagged=True
            )

    for conll_to_pred in to_parse_list:
        parsed_path = os.path.join( parsed_dir, os.path.basename(conll_to_pred) )
        parse_tagged(nlp_dep, conll_to_pred, parsed_path)


## 2. Input is conllu file with gold tokenization ##
def parse_toked(nlp, to_parse_path, parsed_path):
    #to_parse_path is conllu file with gold tokenization
    to_parse_doc = CoNLL.conll2doc(input_file = to_parse_path )
    processed_doc = nlp(to_parse_doc)

    with open(parsed_path, 'w') as f:
        f.write( CoNLL.doc2conll_text(processed_doc) )

def parse_toked_no_lemma(nlp_pos, nlp_dep, to_parse_path, parsed_path ):
    #to_parse_path is conllu file with gold tokenization
    to_parse_doc = CoNLL.conll2doc(input_file = to_parse_path )
    tokpos_doc = nlp_pos( to_parse_doc  )
    parsed_doc = nlp_dep(tokpos_doc)
    with open(parsed_path, 'w') as f:
        f.write( CoNLL.doc2conll_text(parsed_doc) ) 
        

def pred_stanza_toked(to_parse_list, parsed_dir, model_dir, pretrain_path):
    #to_parse_path : pretoked conllu file
    print(pretrain_path)
    to_parse_list = [to_parse_list] if isinstance(to_parse_list, str) else to_parse_list
    # some language such as zh may have _ everywhere in lemma column
    lemma_path = os.path.join(model_dir,'saved_models/lemma/xxx_stanza_lemmatizer.pt')
    
    if os.path.exists(lemma_path):
        nlp_dep = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,pos,lemma,depparse', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            
            lemma_model_path = lemma_path,
            depparse_model_path= os.path.join(model_dir, 'saved_models/depparse/xxx_stanza_parser.pt'),
            depparse_pretrain_path = pretrain_path,
            tokenize_pretokenized=True,
            )
    else:
        nlp_pos = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,mwt,pos', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            tokenize_pretokenized=True,
            )
        
        nlp_dep = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors= 'tokenize,pos,lemma,depparse', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            
            depparse_model_path= os.path.join(model_dir, 'saved_models/depparse/xxx_stanza_parser.pt'),
            depparse_pretrain_path = pretrain_path,
            tokenize_pretokenized=True,
            pretagged=True
            )
        
        for conll_to_pred in to_parse_list:
            parsed_path = os.path.join( parsed_dir, os.path.basename(conll_to_pred) )
            parse_toked_no_lemma(nlp_pos, nlp_dep, conll_to_pred, parsed_path )
        return

    for conll_to_pred in to_parse_list:
        parsed_path = os.path.join( parsed_dir, os.path.basename(conll_to_pred) )
        parse_toked(nlp_dep, conll_to_pred, parsed_path )
        




## 3. Input is raw text file, only use stanza to tokenize it##
def pred_onlytok_stanza(to_tok_list, tokenized_dir, model_dir):
    to_tok_list = [to_tok_list] if isinstance(to_tok_list, str) else to_tok_list
    mwt_path = os.path.join(model_dir,'saved_models/mwt/xxx_stanza_mwt_expander.pt')
    if os.path.exists(mwt_path):
        nlp_tok = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,mwt', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            mwt_model_path = mwt_path, #mwt
        )
    else:
        nlp_tok = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,mwt', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt')
        )
    for to_tok_path in to_tok_list:
        toked_doc = nlp_tok(open(to_tok_path).read())
        tokenized_path = os.path.join( tokenized_dir, os.path.basename(to_tok_path) )
        with open(tokenized_path, 'w') as f:
            f.write( CoNLL.doc2conll_text(toked_doc) ) 


## 4. Input is raw text file and use stanza to parse it ##

def parse_sent_by_sent(nlp, to_parse_path, parsed_path):
    print("Parse sentence by sentence")
    res = []
    to_parse_txt = open(to_parse_path).read().strip().split('\n')
    for sent in to_parse_txt:
        processed_doc = nlp(sent)
        res.append(CoNLL.doc2conll_text(processed_doc))
    with open(parsed_path, 'w') as f:
        f.write( '\n\n'.join( [conll_sent.strip() for conll_sent in res  ] ) + '\n\n' )


def parse_raw(nlp, to_parse_list, parsed_dir, by_sent = False):
    #to_parse_path is txt file
    to_parse_list = [to_parse_list] if isinstance(to_parse_list, str) else to_parse_list

    for to_parse_path in to_parse_list:
        parsed_path = os.path.join( parsed_dir, os.path.basename(to_parse_path) )

        if by_sent:
                parse_sent_by_sent(nlp, to_parse_path, parsed_path)
            #TODO test if all parser trained with 10 sent would take whole doc as one sentence
        else:
            try:
                processed_doc = nlp(open(to_parse_path).read())
                with open(parsed_path, 'w') as f:
                    f.write( CoNLL.doc2conll_text(processed_doc) ) 
            except:
                parse_sent_by_sent(nlp, to_parse_path, parsed_path)       


def parse_sent_by_sent_no_lemma(nlp_pos, nlp_dep, to_parse_path, parsed_path):
    print("Parse sentence by sentence without lemmatizer")
    res = []
    to_parse_txt = open(to_parse_path).read().strip().split('\n')
    for sent in to_parse_txt:
        tokpos_doc = nlp_pos(sent)
        parsed_doc = nlp_dep(tokpos_doc)
        res.append(CoNLL.doc2conll_text(parsed_doc))
    with open(parsed_path, 'w') as f:
        f.write( '\n\n'.join( [conll_sent.strip() for conll_sent in res  ] ) + '\n\n' )

def parse_no_lemma(nlp_pos, nlp_dep, to_parse_list, parsed_dir, by_sent = False ):
    #to_parse_path is txt file
    to_parse_list = [to_parse_list] if isinstance(to_parse_list, str) else to_parse_list

    for to_parse_path in to_parse_list:
        parsed_path = os.path.join( parsed_dir, os.path.basename(to_parse_path) )
        if by_sent:
            parse_sent_by_sent_no_lemma(nlp_pos, nlp_dep, to_parse_path, parsed_path)
        else:
            try:
                tokpos_doc = nlp_pos(open(to_parse_path).read())
                parsed_doc = nlp_dep(tokpos_doc)
                with open(parsed_path, 'w') as f:
                    f.write( CoNLL.doc2conll_text(parsed_doc) ) 
            except:
                parse_sent_by_sent_no_lemma(nlp_pos, nlp_dep, to_parse_path, parsed_path)  



def pred_stanza_raw(to_parse_list, parsed_dir, model_dir, pretrain_path, by_sent = False):
    # todo keep language names that don't in stanza_resources
    print(pretrain_path)
    to_parse_list = [to_parse_list] if isinstance(to_parse_list, str) else to_parse_list
    # some language such as zh may have _ everywhere in lemma column
    lemma_path = os.path.join(model_dir,'saved_models/lemma/xxx_stanza_lemmatizer.pt')
    mwt_path = os.path.join(model_dir,'saved_models/mwt/xxx_stanza_mwt_expander.pt')
    
    if os.path.exists(lemma_path) and os.path.exists(mwt_path):
        nlp_dep = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,mwt,pos,lemma,depparse', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            mwt_model_path = mwt_path, #mwt

            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            
            lemma_model_path = lemma_path, #lemma
            depparse_model_path= os.path.join(model_dir, 'saved_models/depparse/xxx_stanza_parser.pt'),
            depparse_pretrain_path = pretrain_path,
            )
    elif os.path.exists(lemma_path) and not os.path.exists(mwt_path):
        nlp_dep = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize, mwt, pos,lemma,depparse', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),

            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            
            lemma_model_path = lemma_path, #lemma
            depparse_model_path= os.path.join(model_dir, 'saved_models/depparse/xxx_stanza_parser.pt'),
            depparse_pretrain_path = pretrain_path,
            )
    elif not os.path.exists(lemma_path) and os.path.exists(mwt_path):
        print("NO LEMMA BUT HAS MWT")
        nlp_dep = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize, mwt, pos,lemma,depparse', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            mwt_model_path = mwt_path, #mwt

            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            
            depparse_model_path= os.path.join(model_dir, 'saved_models/depparse/xxx_stanza_parser.pt'),
            depparse_pretrain_path = pretrain_path,
            )
    else:
        #no lemma and no multi word token
        nlp_pos = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,mwt,pos', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            )
        
        nlp_dep = stanza.Pipeline(
            lang='xxx',
            download_method = stanza.pipeline.core.DownloadMethod.REUSE_RESOURCES,
            processors='tokenize,mwt, pos,depparse', 
            tokenize_model_path = os.path.join(model_dir, 'saved_models/tokenize/xxx_stanza_tokenizer.pt'),
            pos_model_path = os.path.join(model_dir, 'saved_models/pos/xxx_stanza_tagger.pt'),
            pos_pretrain_path = pretrain_path,
            
            depparse_model_path= os.path.join(model_dir, 'saved_models/depparse/xxx_stanza_parser.pt'),
            depparse_pretrain_path = pretrain_path,
            tokenize_pretokenized=True,
            pretagged=True
            )
        for to_parse_path in to_parse_list:
            parsed_path = os.path.join( parsed_dir, os.path.basename(to_parse_path) )
            parse_no_lemma(nlp_pos, nlp_dep, to_parse_path, parsed_path, by_sent )
        return 

    for to_parse_path in to_parse_list:
        parsed_path = os.path.join( parsed_dir, os.path.basename(to_parse_path) )
        parse_raw(nlp_dep, to_parse_path, parsed_path, by_sent)
    


def train_pred_stanza(project_path, parser_id, epochs = 5, need_train = True, keep_pos = True, tokenized = True,  epochs_tok = 100):
    print("stanza train pred")
    res_folder = os.path.join( abs_root, project_path, f"{parser_id}_res")
    conll_path = os.path.join( res_folder, 'extern_data/conllus/xxx_stanza')

    #TODO the case that to_parse_files are txt
    input_path = os.path.join(abs_root, project_path, 'input_conllus')
    to_pred_path = os.path.join(  input_path, 'to_parse') 

    to_parse_names = open( os.path.join( abs_root, project_path, TO_PARSE_NAMES)).read().strip().split('\t')
    # to_parse_list = [ os.path.join( to_pred_path, f)  for f in os.listdir(to_pred_path) if f[-7:] == '.conllu' ]  
    to_parse_list = [ os.path.join( to_pred_path, f + '.conllu') for f in to_parse_names ]
    train_list = [ os.path.join( input_path, f) for f in os.listdir(input_path) if f[-7:] == '.conllu']

    predicted_path = os.path.join( res_folder,  "predicted/") 
    Path( predicted_path).mkdir(parents=True, exist_ok=True)

    os.chdir(res_folder)
    os.system("pwd")
    
    # Train  
    if need_train:
        #make pretrain wv
        #TODO the case that to_parse_files are txt
        logging( os.path.join( abs_root, project_path), f'Pretrain embedding with FastText\n')
        emb_train_list = to_parse_list + train_list
        pretrain_path = pretrain_wv( emb_train_list, wv_dir = "data/wordvec/fasttext/Custom_Lang", vec_size = 100)
        #TODO 11/08/  22:06
        k = len( open( os.path.join( conll_path, 'xxx_stanza-ud-train.conllu' )).read().strip().split('\n\n') ) 
        train_stanza(res_folder, k, pretrain_path = pretrain_path, with_gold = keep_pos, batch_size = 8,  batch_size_tok = 8, epochs = epochs, epochs_tok = epochs_tok)
        
    #Parse
    to_parse_all = [ os.path.join( to_pred_path, fname ) for fname in to_parse_list ] 
    to_parse_all += [ os.path.join( input_path, fname ) for fname in train_list ] 

    eval_path = os.path.join( res_folder, EVAL_DEV_NAME)
    Path( eval_path ).mkdir(parents=True, exist_ok=True)
    dev_path = os.path.join( conll_path, 'xxx_stanza-ud-dev.conllu')
    logging( os.path.join( abs_root, project_path), f'Parsing files...\n')


    if keep_pos: #pretagged
        pred_stanza_tagged(to_parse_all,  parsed_dir = predicted_path, model_dir = res_folder, pretrain_path = pretrain_path)
        #dev
        pred_stanza_tagged([dev_path],  parsed_dir = eval_path, model_dir = res_folder, pretrain_path = pretrain_path)
        return predicted_path
    if tokenized:
        pred_stanza_toked(to_parse_all, predicted_path, res_folder, pretrain_path)
        #dev
        pred_stanza_toked([dev_path],  parsed_dir = eval_path, model_dir = res_folder, pretrain_path = pretrain_path)
    else:
        pred_stanza_raw(to_parse_all, predicted_path, res_folder, pretrain_path, by_sent = False)
        #dev
        pred_stanza_raw([dev_path],  parsed_dir = eval_path, model_dir = res_folder, pretrain_path = pretrain_path, by_sent = False)
    # os.system(f"mv {os.path.join(eval_path, 'xxx_stanza-ud-dev.conllu' )}  {os.path.join(eval_path, 'dev.conllu')}")
    
    return predicted_path




if __name__ == '__main__':
    if len(sys.argv)  < 8:
        print("Usage: train_pred_stanza.py project_path parser_id need_train epochs epochs_tok keep_pos tokenized", file=sys.stderr)
        sys.exit(-1)

    project_path = sys.argv[1]
    parser_id  = sys.argv[2]
    model_dir = os.path.join(project_path, f'{parser_id}_res' )

    need_train = sys.argv[3]
    print("Stanza:", type(need_train), need_train)
    epochs = int(sys.argv[4])
    epochs_tok = int(sys.argv[5])
    keep_pos = sys.argv[6]
    tokenized = sys.argv[7]
    print("keep pos:", type(keep_pos), keep_pos)
    print("tokenized:", type(tokenized), tokenized)

    #!!!Important to change word directory as model_dir e.g. stanza_tok_res that including save_models, data, extren_datat etc.
    os.chdir(model_dir)
    os.system("pwd")

    train_pred_stanza(project_path, parser_id, epochs = epochs, need_train = need_train, \
        keep_pos = keep_pos, tokenized = tokenized,  epochs_tok = epochs_tok)

    # try:
    #     train_pred_stanza(project_path, parser_id, epochs = epochs, need_train = need_train, \
    #         keep_pos = keep_pos, tokenized = tokenized,  epochs_tok = epochs_tok)
    # except FileNotFoundError:
    #     logging(os.path.join( abs_root, project_path),   'Error\n')
    #     remove_project( os.path.join( abs_root, project_path))

