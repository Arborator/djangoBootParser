#data process for djangoBootParser
import json, zipfile, os, sys, time,random,re
import numpy as np
from pathlib import Path
import base64, hashlib 


#global var for djangoBootParser
PROJ_ALL_PATH = 'projects/'
TO_PARSE_NAMES = 'to_parse_fnames.tsv'
#log path and parsed path to modified according to project 
LOG_NAME = 'progress.txt'
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)
sid_pattern = re.compile(r'# sent_id =(.+)')
uid_pattern = re.compile(r'# user_id =(.+)')
time_pattern = re.compile(r'# timestamp =(.+)')
comment_pattern = re.compile( r'# .+' )
comment_pattern_tosub = re.compile( r'# .+\n' )

#For every parser
def make_data_dict(conllu_str, uid_to_add = 'hopsParser' ): 
    sent_ls = conllu_str.strip().split('\n\n')
    data_dict = {}
    
    add_sid_ls = False
    for idx, sent in enumerate(sent_ls):
        sid = re.findall(sid_pattern, sent)
        assert(len(sid) <=1 )
        data_dict[idx] = {'sent_id': sid[0]} if sid else {'sent_id': idx}
        
        if '# user_id =' in sent and '# timestamp =' in sent:
            sent = re.sub( uid_pattern, f'# user_id = {uid_to_add}', sent)
            sent = re.sub( time_pattern, '# timestamp = 0', sent)
            data_dict[idx]['comment'] = re.findall(comment_pattern, sent)
        elif '# user_id =' in sent and '# timestamp =' not in sent:
            sent = re.sub( uid_pattern, f'# user_id = {uid_to_add}', sent)
            data_dict[idx]['comment'] = re.findall(comment_pattern, sent)
            data_dict[idx]['comment'].append('# timestamp = 0')
        elif '# user_id =' not in sent and '# timestamp =' in sent:
            sent = re.sub( time_pattern, '# timestamp = 0', sent)
            data_dict[idx]['comment'] = re.findall(comment_pattern, sent)
            data_dict[idx]['comment'].append(f'# user_id = {uid_to_add}')
        else:
            data_dict[idx]['comment'] = re.findall(comment_pattern, sent)
            data_dict[idx]['comment'].append(f'# user_id = {uid_to_add}')
            data_dict[idx]['comment'].append('# timestamp = 0')
        
        if sid == []:
            # print("Adding sent id")
            add_sid_ls = True
            data_dict[idx]['comment'].append(f"# sent_id = {idx}")

        conllu = re.sub(comment_pattern, '' , sent).strip().split('\n')
        data_dict[idx]['conllu'] = [tok.split('\t') for tok in conllu ]
    if add_sid_ls:
        print("Added sent id")
    return data_dict
        

def replace_col(dict_in, dict_out, to_replace_list, repl_comment = False):
    # print("DEBUG REPLACE: ", to_replace_list)
    assert(len(dict_in) == len(dict_out) )
    for idx, sent in dict_out.items():
        for tid in range(len(sent['conllu'])):
            assert(len(dict_out[idx]['conllu'][tid]) == 10)
            assert(len(dict_in[idx]['conllu'][tid]) == 10)
            for cp in to_replace_list:
                dict_out[idx]['conllu'][tid][cp] = dict_in[idx]['conllu'][tid][cp]
            if repl_comment:
                dict_out[idx]['comment'] = dict_in[idx]['comment']
    # print("DICT OUT O: ", dict_out[0])
    return dict_out


def dict2conll(dict_conll):
    sent_ls = []
    for idx, sent in dict_conll.items():
        to_write_conll = sent['comment'] + [ '\t'.join(tok) for tok in sent['conllu']]
        sent_ls.append( '\n'.join(to_write_conll))
    conll_str = '\n\n'.join(sent_ls)+'\n\n'
    return conll_str


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




# For kirian's parser 
def add_multitok(parsed_path, gold_path):
    parsed = [sent.split('\n') for sent in open(parsed_path).read().strip().split('\n\n')]
    subpattern = re.compile(r'# .*\n')
    gold = [sent.split('\n') for sent in re.sub(subpattern, '', open(gold_path).read().strip()).split('\n\n')]

    has_multi = False

    for sid, sent in enumerate(parsed):
        to_add = []
        comment = 0
        for tid,tok in enumerate(sent):
            if tok[0] == '#':
                comment += 1
                continue

            gold_split = gold[sid][tid - comment].split('\t', 2)[:2]
            if tok.split('\t', 2)[:2] != gold_split:
                has_multi = True
                #print('multi word token :', gold[sid][tid - comment])
                assert('-' in gold_split[0])
                sent[:] = sent[: tid] + [gold[sid][tid - comment]] + sent[tid:] #deep copy
        parsed[sid] = sent

    if has_multi:
        print('adding lines of multi word from gold')
        with open(parsed_path, 'w') as f:
            f.write( '\n\n'.join( ['\n'.join(sent) for sent in parsed ]  ) + '\n\n' )


# to adapte, for trankit
def copy_lemma_file(lemma_path, posdep_path):
    # TODO
    #better to combine in backend when we copy the upos???
    print('copy lemma:', lemma_path)
    lemma_txt  = open(lemma_path).read().strip()
    begin, tmp = lemma_txt.split("sent_id ", 1)
    lemmas= [t.split('\n') for t in ("# sent_id "+tmp).split('\n\n') if t]

    lemma_dict = {}
    for conllu in lemmas:
        # every sent begin with #sent_id 
        # TODO replace this by keyword sent_id instead of index 
        key = conllu[0].split('=')[1].strip()
        lemma_dict[key] = [line for line in conllu[1:] if line[0] != '#']

    posdep_txt = open(posdep_path).read().strip()
    begin, tmp = parsed_txt.split("sent_id ", 1)
    deprel = [t.split('\n') for t in ("# sent_id "+tmp).split('\n\n') if t]

    posdep_dict = {}
    for conllu in deprel:
        key = conllu[0].split('=')[1].strip()
        posdep_dict[key] = conllu[1:]

    for key, conll in posdep_dict.items():
        begin = 0
        for l, line in enumerate(conll):
            if(line[0]!='#'):
                info = line.split('\t')
                info_tag = lemma_dict[key][l - begin].split('\t')
                #print(info)
                info[3] = info_tag[LEMMA]
                posdep_dict[key][l] = '\t'.join(info)
            else:
                begin += 1 
        posdep_dict[key] = '\n'.join(posdep_dict[key])

    to_write = begin[:-2] + '\n\n'.join([f'# sent_id = {k}\n' + val for k, val in posdep_dict.items()]) + '\n\n'
    with open(os.path.join(os.path.dirname(posdep_path), 'combined_parsed.conllu'), 'w' ) as f:
        f.write(to_write)




