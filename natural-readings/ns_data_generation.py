import pandas as pd
from multiprocessing import Pool
import spacy
import json
import numpy as np
import collections
from metrics import *


import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)



nlp = spacy.load("en_core_web_sm")

# Frequency dictionary, from Google Unigram, case insensitive
with open('freq-dict.json') as json_file: 
    freq_dict = json.load(json_file) 

all_heads = [(i,j) for i in range(12) for j in range(12)]
top_syntactic_heads =[(0,6),(1,0),(1,1),(2,0),(2,3),(2,8),(2,9),(3,5),(3,6),(3,8),(3,9),(3,11),(4,0),(4,2),(4,3),(4,9),(4,11),(6,7),(7,8),(10,5)]
not_top_syntactic_heads = [(i,j) for i in range(12) for j in range(12) if (i,j) not in top_syntactic_heads]
    
def __main__():
    NS_RT = pd.read_csv('ns-raw-data.tsv', delimiter= '\t')
    # Iterate over unique combinations of 'item' and 'zone'
    
    for (item_value, zone_value), group in NS_RT.groupby(['item', 'zone']):
        # Find the unique sentID for each (item, zone) combination
        unique_sentID = np.unique(group['sentID'])[0]
        if unique_sentID==None:
            break
        # Assign the unique sentID back to the DataFrame for the respective (item, zone) combination
        NS_RT.loc[(NS_RT['item'] == item_value) & (NS_RT['zone'] == zone_value), 'sentID'] = unique_sentID


    # Drop duplicates to get unique combinations of 'item' and 'sentID'
    unique_combinations = NS_RT[['item', 'sentID']].dropna(subset=['item', 'sentID']).drop_duplicates().values

    # Iterate over each unique combination
    for item_value, sentID in unique_combinations:
 
        filtered_corpus =NS_RT.loc[(NS_RT['item'] == item_value) & (NS_RT['sentID'] == sentID)]
        words_by_zone = filtered_corpus.groupby('zone', as_index=False).first()[['item','sentID','zone', 'word']]
        word_list = list(words_by_zone.word)
        sent = ' '.join(words_by_zone['word'])
        if sentID ==1: context = "" 
        else: sent = f' {sent}'

        context_len = len(tokenizer.encode(context)) if len(context) >0 else 0
        surprisals = get_surprisal(context, sent) # get the surprisals for the input sentence, given preceding context
        top_syn_entropies_30 = get_entropy_from_sets(context, sent, 30, top_syntactic_heads)
        not_top_syn_entropies_30 = get_entropy_from_sets(context, sent, 30, not_top_syntactic_heads)
        all_entropies_30 = get_entropy_from_sets(context, sent, 30, all_heads)
        duplicate_token_dict = collections.defaultdict(list)


        doc = nlp(sent)
        spacy_info_dict = collections.defaultdict(list)
        
        for t in doc:
            spacy_info_dict[t.text].append((t.pos_, t.tag_, t.dep_, t.i, t.head.i))
        idx = 0
    
        for i, word in enumerate(words_by_zone['word']):
            context_len = min(context_len, 1000)    
            zone = words_by_zone['zone'].iloc[i]
            zone_idx = NS_RT.loc[NS_RT["item"]==item_value].loc[NS_RT['zone']==zone].index        
            found_tokens = find_token_position(word_list,word)
        
            if word not in duplicate_token_dict: duplicate_token_dict[word] = found_tokens

            if len(spacy_info_dict[word]) != word_list[i:].count(word):
                crnt_pos, crnt_tag, crnt_dep, token_loc, head_loc = None, None, None, None, None
            else: 
                crnt_pos, crnt_tag, crnt_dep, token_loc, head_loc = spacy_info_dict[word][0]
                spacy_info_dict[word] = spacy_info_dict[word][1:]

            duplicate_token_dict[word] = duplicate_token_dict[word][1:]   
            token_len = len(tokenizer.tokenize(f' {word}'))
            surprisal_set = [i for i in surprisals[idx:idx+token_len] if i != None]

            crnt_surprisal = max(surprisal_set) if surprisal_set else None
            crnt_top_syn_entropy_30 =  max(top_syn_entropies_30[idx:idx+token_len])
            crnt_not_top_syn_entropy_30 =  max(not_top_syn_entropies_30[idx:idx+token_len])
            crnt_all_entropy_30 =  max(all_entropies_30[idx:idx+token_len])
            crnt_entropy_dict = {}
            

            crnt_freq = round(np.log10(freq_dict[word.lower()]),4) if word.lower() in freq_dict else None
            crnt_w_len = len(word)

            NS_RT.loc[zone_idx, 'num_tokens'] = token_len
            NS_RT.loc[zone_idx, 'surprisal'] = crnt_surprisal
            NS_RT.loc[zone_idx, 'top_syn_entropy_30'] =  crnt_top_syn_entropy_30
            NS_RT.loc[zone_idx, 'not_top_syn_entropy_30'] =  crnt_not_top_syn_entropy_30
            NS_RT.loc[zone_idx, 'all_entropy_30'] =  crnt_all_entropy_30
            NS_RT.loc[zone_idx, 'pos'] =  crnt_pos
            NS_RT.loc[zone_idx, 'tag'] =  crnt_tag
            NS_RT.loc[zone_idx, 'dep'] =  crnt_dep
            NS_RT.loc[zone_idx, 'rel_distance'] =  token_loc-head_loc if token_loc!=None else None
            NS_RT.loc[zone_idx, 'frequency'] =  crnt_freq
            NS_RT.loc[zone_idx, 'position'] =  i+1
            NS_RT.loc[zone_idx, 'w_len'] =  crnt_w_len

            if len(context) > 0:

                NS_RT.loc[zone_idx, 'prev_surprisal'] = prev_surprisal
                NS_RT.loc[zone_idx, 'prev_top_syn_entropy_30'] = prev_top_syn_entropy_30
                NS_RT.loc[zone_idx, 'prev_not_top_syn_entropy_30'] = prev_not_top_syn_entropy_30
                NS_RT.loc[zone_idx, 'prev_all_entropy_30'] = prev_all_entropy_30
                NS_RT.loc[zone_idx, 'prev_pos']  = prev_pos
                NS_RT.loc[zone_idx, 'prev_tag']  = prev_tag
                NS_RT.loc[zone_idx, 'prev_dep']  = prev_dep
                NS_RT.loc[zone_idx, 'prev_frequency'] = prev_freq
                NS_RT.loc[zone_idx, 'prev_w_len']= prev_w_len

            prev_surprisal = crnt_surprisal
            prev_top_syn_entropy_30 =crnt_top_syn_entropy_30
            prev_not_top_syn_entropy_30 =crnt_not_top_syn_entropy_30
            prev_all_entropy_30=crnt_all_entropy_30

            prev_pos = crnt_pos
            prev_tag = crnt_tag
            prev_dep = crnt_dep
            prev_freq = crnt_freq
            prev_w_len = crnt_w_len
            idx += token_len
            context_len += token_len
    
        context += sent

        # Set indices for both dataframes to align on the keys
        NS_RT.set_index(['item', 'sentID', 'zone', 'word'], inplace=True)
        words_by_zone.set_index(['item', 'sentID', 'zone', 'word'], inplace=True)
        # Update corpus with matching values from words_y_zone
        NS_RT.update(words_by_zone)
        # Reset index if necessary
        NS_RT.reset_index(inplace=True)


    NS_RT.to_csv('ns-annotated-data-all.csv')


if __name__ == "__main__":
    __main__()
        
