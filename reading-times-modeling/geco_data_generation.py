import pandas as pd
from multiprocessing import Pool
import spacy
import json
import numpy as np
import collections
from metrics import *

nlp = spacy.load("en_core_web_sm")

# Frequency dictionary, from Google Unigram, case insensitive
with open('freq-dict.json') as json_file: 
    freq_dict = json.load(json_file) 

top_syntactic_heads =[(0,6),(1,0),(1,1),(2,0),(2,3),(2,8),(2,9),(3,5),(3,6),(3,8),(3,9),(3,11),(4,0),(4,2),(4,3),(4,9),(4,11),(6,7),(7,8),(10,5)]
not_top_syntactic_heads = [(i,j) for i in range(12) for j in range(12) if (i,j) not in top_syntactic_heads]

# Initialize an empty dictionary to store new columns before updating Geco_RT
new_columns = collections.defaultdict(list)

# Loop through each unique sentID
def __main__():
    Geco_RT = pd.read_csv('geco-raw-data.csv')
    unique_sentID = list(np.unique(Geco_RT['sent_key']))
    prev_pos = None
    prev_tag = None
    prev_dep = None
    prev_freq = None
    prev_w_len = None
    prev_surprisal = None
    prev_top_syn_entropies_30 = None
    prev_not_top_syn_entropies_30 = None

    for sentID in unique_sentID:
        # Filter corpus by sent_key
        filtered_corpus = Geco_RT.loc[Geco_RT['sent_key'] == sentID]
        words_by_key = filtered_corpus.loc[filtered_corpus['PP_NR'] == filtered_corpus['PP_NR'].iloc[0]][['sent_key', 'WORD_ID', 'WORD']]
        word_list = list(words_by_key['WORD'])
        word_list = [i if isinstance(i, str) else "null" for i in word_list ]
        sent = ' '.join(word_list)
        print(sentID, sent)
        if sentID == 0: context = ""
        else: sent = f' {sent}'

        surprisals = get_surprisal(context, sent) # get the surprisals for the input sentence, given preceding context
        top_syn_entropies_30 = get_entropy_from_sets(context, sent,  window_size=30, headsets = top_syntactic_heads)
        not_top_syn_entropies_30 = get_entropy_from_sets(context, sent,  window_size=30, headsets = not_top_syntactic_heads)

        duplicate_token_dict = collections.defaultdict(list)
        doc = nlp(sent)
        spacy_info_dict = collections.defaultdict(list)

        # Process tokens and store their information in spacy_info_dict
        for t in doc:
            spacy_info_dict[t.text].append((t.pos_, t.tag_, t.dep_, t.i, t.head.i))

        idx = 0
        for i, word in enumerate(word_list):
 
            # key = words_by_key['WORD_ID'].iloc[i]
            # key_idx = Geco_RT.loc[Geco_RT['WORD_ID'] == key].index
            found_tokens = find_token_position(word_list, word)

            # Collect token positions in duplicate_token_dict
            if word not in duplicate_token_dict: duplicate_token_dict[word] = found_tokens
        
            # Default values for token information
            crnt_pos, crnt_tag, crnt_dep = None, None, None
        
            # Ensure that the spacy information matches the expected frequency
            if len(spacy_info_dict[word]) != word_list[i:].count(word):
                pass
            else:
                crnt_pos, crnt_tag, crnt_dep ,_ , _ = spacy_info_dict[word][0]
                spacy_info_dict[word] = spacy_info_dict[word][1:]

            # Update the duplicate token dictionary for word
            duplicate_token_dict[word] = duplicate_token_dict[word][1:]   

            token_len = len(tokenizer.tokenize(f' {word}'))
            crnt_w_len = len(word)
            surprisal_set = [i for i in surprisals[idx:idx+token_len] if i != None]
            top_syn_entropies_30_set = [i for i in top_syn_entropies_30[idx:idx+token_len] if i != None]
            not_top_syn_entropies_30_set = [i for i in not_top_syn_entropies_30[idx:idx+token_len] if i != None]
            crnt_surprisal = max(surprisal_set) if surprisal_set else None
            crnt_top_syn_entropies_30 = max(top_syn_entropies_30_set) if surprisal_set else None
            crnt_not_top_syn_entropies_30 = max(not_top_syn_entropies_30_set) if surprisal_set else None
            crnt_freq = round(np.log10(freq_dict[word.lower()]), 4) if word.lower() in freq_dict else None

            # Store token-level information into new_columns
            new_columns['num_tokens'].append(token_len)
            new_columns['pos'].append(crnt_pos)
            new_columns['tag'].append(crnt_tag)
            new_columns['dep'].append(crnt_dep)
            new_columns['frequency'].append(crnt_freq)
            new_columns['position'].append(i + 1)
            new_columns['w_len'].append(crnt_w_len)
            new_columns['crnt_surprisal'].append(crnt_surprisal)
            new_columns['crnt_top_syn_entropies_30'].append(crnt_top_syn_entropies_30)
            new_columns['crnt_not_top_syn_entropies_30'].append(crnt_not_top_syn_entropies_30)

            new_columns['prev_pos'].append(prev_pos)
            new_columns['prev_tag'].append(prev_tag)
            new_columns['prev_dep'].append(prev_dep)
            new_columns['prev_frequency'].append(prev_freq)
            new_columns['prev_w_len'].append(prev_w_len)
            new_columns['prev_surprisal'].append(prev_surprisal)
            new_columns['prev_top_syn_entropies_30'].append(prev_top_syn_entropies_30)
            new_columns['prev_not_top_syn_entropies_30'].append(prev_not_top_syn_entropies_30)
        
            prev_pos = crnt_pos
            prev_tag = crnt_tag
            prev_dep = crnt_dep
            prev_freq = crnt_freq
            prev_w_len = crnt_w_len
            prev_surprisal = crnt_surprisal
            prev_top_syn_entropies_30 = crnt_top_syn_entropies_30
            prev_not_top_syn_entropies_30 = crnt_not_top_syn_entropies_30
            idx += token_len

        context += sent

    # After processing all sentences, create a new DataFrame with the collected columns
    new_data = pd.DataFrame(new_columns)

    # Use pd.concat() to add new columns to Geco_RT
    Geco_RT = pd.concat([Geco_RT, new_data], axis=1)

    # reset the index
    Geco_RT.reset_index(inplace=True)

    columns_to_update = ['WORD','num_tokens', 'pos', 'tag', 'dep', 
                     'frequency', 'position', 'w_len', 'crnt_surprisal', 
                     'crnt_top_syn_entropies_30', 'crnt_not_top_syn_entropies_30', 
                     'prev_pos', 'prev_tag', 'prev_dep', 'prev_frequency', 
                     'prev_w_len', 'prev_surprisal', 'prev_top_syn_entropies_30', 
                     'prev_not_top_syn_entropies_30']

    for unique_id in np.unique(Geco_RT['WORD_ID']):
        print(unique_id)
        # Select the subset of rows based on the condition
        subset = Geco_RT[Geco_RT['WORD_ID'] == unique_id][columns_to_update]

        # Get the values from the first row of the subset
        first_row_values = subset.iloc[0]

        # Assign these values to all rows in the subset
        Geco_RT.loc[Geco_RT['WORD_ID'] == unique_id, columns_to_update] = first_row_values.tolist()

    Geco_RT.to_csv('geco-annotated-data.csv')

if __name__ == "__main__":
    __main__()
