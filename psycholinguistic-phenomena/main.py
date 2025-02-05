import pandas as pd
import numpy as np
from metrics import *
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


top_syntactic_heads =[(0,6),(1,0),(1,1),(2,0),(2,3),(2,8),(2,9),(3,5),(3,6),(3,8),(3,9),(3,11),(4,0),(4,2),(4,3),(4,9),(4,11),(6,7),(7,8),(10,5)]
not_top_syntactic_heads = [(i,j) for i in range(12) for j in range(12) if (i,j) not in top_syntactic_heads]
all_heads = [(l,h) for l in range(12) for h in range(12)]

def get_center_embedding_data():

    Stolz = pd.read_csv('data/Stolz1967.csv')
    df = pd.DataFrame()

    for sent_id in range(1,16):
        id_set = Stolz.loc[Stolz['id']==sent_id]
        lv1noun, lv2noun, lv3noun = id_set.iloc[0].subj1, id_set.iloc[0].subj2, id_set.iloc[0].subj3
        lv1verb, lv2verb, lv3verb = id_set.iloc[0].verb1, id_set.iloc[0].verb2, id_set.iloc[0].verb3

        for lv in [1,2,3]:
            this_level = id_set[id_set['level']==lv]

            for sent_type in ["CE","RB"]:
                this_sent_type = this_level[this_level['type']==sent_type]
                sentence = this_sent_type.iloc[0].sentence
                indexed_tokens = tokenizer.encode(sentence)
                tokens = tokenizer.convert_ids_to_tokens(indexed_tokens)
                tokens = [tokenizer.decode(i).strip() for i in indexed_tokens]           
            
                if lv == 1: source, target = lv1verb, lv1noun
                elif lv == 2: source, target = lv2verb, lv2noun 
                elif lv == 3: source, target = lv3verb, lv3noun

                if len(find_token_position(tokens, source))!=1:
                    break
            
                source_loc = find_token_position(tokens, source)[0]
                surprisal = get_surprisal(None, sentence)[source_loc] # This line of code was incorrect in the original code. Need to be careful.
                entropy_43 = get_entropy_from_sets(None, sentence, headsets=[(4,3)])[source_loc]
                entropy_syntactic = get_entropy_from_sets(None, sentence,headsets=top_syntactic_heads)[source_loc]
                entropy_global = get_entropy_from_sets(None, sentence,headsets=all_heads)[source_loc]


                attn_to_target = get_paid_attention(sentence, source, target, head = [4,3])
                attn_to_lv1_noun = get_paid_attention(sentence, source, lv1noun, head = [4,3])
                attn_to_lv2_noun = get_paid_attention(sentence, source, lv2noun, head = [4,3])
                attn_to_lv3_noun = get_paid_attention(sentence, source, lv3noun, head = [4,3])

                newRow = pd.DataFrame({'id':[sent_id],
                                    "type":[sent_type],
                                    "level":[lv],
                                    "verb":[source],
                                    "sentence":[sentence],
                                    "target":[target],
                                    "surprisal":[surprisal],
                                    "entropy_43": [entropy_43],
                                    "entropy_syntactic": [entropy_syntactic], 
                                    "entropy_global": [entropy_global], 
                                    "attn_to_target":[attn_to_target],  
                                    "attn_to_lv1_noun": [attn_to_lv1_noun],
                                    "attn_to_lv2_noun": [attn_to_lv2_noun],
                                    "attn_to_lv3_noun": [attn_to_lv3_noun]} )
                df = pd.concat([df, newRow],ignore_index=True) 
    df.to_csv('results/stolz-results-new.csv')

def get_relative_clause_data():
    Staub_rcv = pd.read_csv('data/Staub2010_loc_control_orc.csv')
    Staub_rcn = pd.read_csv('data/Staub2010_loc_control_src.csv')
    sent_type = ["SR","OR"]

    df = pd.DataFrame()
    for i in range(1,25):

        rcv_set = Staub_rcv.loc[Staub_rcv['id']==i]
        rcn_set = Staub_rcn.loc[Staub_rcn['id']==i]

        for t in sent_type:

            ## Add rcv data
            rcv_sent = rcv_set.loc[rcv_set['type']==t]
            sentence = rcv_sent['sentence'].iloc[0]
            indexed_tokens = tokenizer.encode(sentence)
            tokens = tokenizer.convert_ids_to_tokens(indexed_tokens)
            tokens = [tokenizer.decode(i).strip() for i in indexed_tokens]        
            source = rcv_sent['crit'].iloc[0] 
            token_positions = find_token_position(tokens, source)  

            if len(token_positions)==1:
                token_position = token_positions[0]
                rcv_surprisal, rcv_entropy_syntactic, rcv_entropy_global = get_surprisal(None, sentence)[token_position], \
                    get_entropy_from_sets(None, sentence, window_size=None, headsets=top_syntactic_heads)[token_position], \
                    get_entropy_from_sets(None, sentence, window_size=None, headsets=all_heads)[token_position] 
            else:        
                rcv_surprisal, rcv_entropy_syntactic,  rcv_entropy_global = None, None, None
            newRow = pd.DataFrame({'id':[i],
                                    "sent_type":[t+"C"],
                                    "sentence":[sentence],
                                    "surprisal":[rcv_surprisal],
                                    "entropy_syntactic": [rcv_entropy_syntactic],
                                    "entropy_global": [rcv_entropy_global],
                                    "type": "rcv"})  
            df = pd.concat([df, newRow],ignore_index=True)    


            ## Add rcn data
            rcn_sent = rcn_set.loc[rcn_set['type']==t]
            sentence = rcn_sent['sentence'].iloc[0]
            indexed_tokens = tokenizer.encode(sentence)
            tokens = tokenizer.convert_ids_to_tokens(indexed_tokens)
            tokens = [tokenizer.decode(i).strip() for i in indexed_tokens]        
            source = rcn_set['crit'].iloc[0] 

            token_position = find_token_position(tokens, 'the')[1]
            rcn_surprisal, rcn_entropy_syntactic, rcn_entropy_global= get_surprisal(None, sentence)[token_position], \
                                             get_entropy_from_sets(None, sentence, window_size=None, headsets=top_syntactic_heads)[token_position], \
                                             get_entropy_from_sets(None, sentence, window_size=None, headsets=all_heads)[token_position] 

            #rcn_surprisal, rcn_entropy_all = None, None
            newRow = pd.DataFrame({'id':[i],
                                "sent_type":[t+"C"],
                                "sentence":[sentence],
                                "surprisal":[rcn_surprisal],
                                "entropy_syntactic": [rcn_entropy_syntactic],
                                "entropy_global": [rcn_entropy_global],
                                "type": "rcn"} )
                
            df = pd.concat([df, newRow],ignore_index=True) 
         
    df.to_csv('results/staub-results-new.csv')

def main():
    get_center_embedding_data()
    print("Data for center embedding vs. right branching sentences are generated.")
    get_relative_clause_data()
    print("Data for relative clause sentences are generated.")

if __name__ == "__main__":
    main()

