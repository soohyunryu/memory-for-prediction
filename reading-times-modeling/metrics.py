from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from multiprocessing import Pool
import scipy.stats
import torch


model = GPT2LMHeadModel.from_pretrained('gpt2',return_dict_in_generate=True, output_attentions = True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_surprisal(context, sent):
    sent_len = len(tokenizer(sent, return_tensors='pt')['input_ids'][0])
    if (context != None) and (len(context) >0):
        input_tokens = tokenizer.encode(context + sent, return_tensors='pt')
        input_tokens = input_tokens[:, -1024:]
    else:
        input_tokens = tokenizer.encode(sent, return_tensors='pt')

    with torch.no_grad():
        # Get logits for each token in the sequence
        outputs = model(input_tokens)
        logits = outputs.logits  # Shape: (1, sequence_length, vocab_size)

    # Calculate probabilities and log probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.log2(probs)
    
    # Gather the log probabilities corresponding to the actual next tokens in the sentence
    target_probs = log_probs[0, torch.arange(len(input_tokens[0]) - 1), input_tokens[0, 1:]]
    
    # Convert to surprisal by taking negative log probability
    surprisal_values = -target_probs
    surprisal_values = surprisal_values.tolist()
    surprisal_values = [round(float(s), 4) for s in surprisal_values]
    
    # Add None for the first token
    if context ==None or len(context)==0:
        surprisal_values.insert(0, None)
    return surprisal_values[-sent_len:]


def get_global_entropy(context, sent, window_size=None):

    if (context != None) and (len(context) >0):
        input_tokens = tokenizer.encode(context + sent, return_tensors='pt')
        input_tokens = input_tokens[:, -1024:]
    else:
        input_tokens = tokenizer.encode(sent, return_tensors='pt')
    sent_len = len(tokenizer(sent, return_tensors='pt')['input_ids'][0])

    entropies_all = []
    with torch.no_grad():
        outputs = model(input_tokens)
        attn = outputs.attentions
    attn  = torch.stack([layer[0] for layer in attn])
    indices = torch.arange(attn.shape[-1])
    attn[:, :, indices, indices] = 0

    if window_size == None:
        for l in range(attn.shape[0]):
            for h in range(attn.shape[1]):
                entropy_vals =[]
                for w in range(attn.shape[2]):
                    entropy_val = scipy.stats.entropy(attn[l][h][w],base=2)
                    entropy_vals.append(entropy_val if entropy_val>0 else 0)
                entropies_all.append(entropy_vals)
    else:
        attn = attn[:,:,-sent_len:,:]
        for l in range(attn.shape[0]):
            for h in range(attn.shape[1]):
                entropy_vals =[]
                for w in range(attn.shape[2]):
                    entropy_val = scipy.stats.entropy(attn[l][h][w][:-(sent_len-w)][-window_size:],base=2)
                    entropy_vals.append(entropy_val if entropy_val>0 else 0)
                entropies_all.append(entropy_vals)

    entropies_all_tensor = torch.tensor(entropies_all)
    global_entropy = torch.mean(entropies_all_tensor, dim=(0))
    global_entropy_rounded_floats = [round(float(val),4) for val in global_entropy]
    return global_entropy_rounded_floats


def get_entropy_from_sets(context, sent, window_size= None, headsets = None):
    if (context != None) and (len(context) >0):
        input_tokens = tokenizer.encode(context + sent, return_tensors='pt')
        input_tokens = input_tokens[:, -1024:]
    else:
        input_tokens = tokenizer.encode(sent, return_tensors='pt')
    sent_len = len(tokenizer(sent, return_tensors='pt')['input_ids'][0])

    entropies_all = []
    with torch.no_grad():
        outputs = model(input_tokens)
        attn = outputs.attentions
    
    attn  = torch.stack([layer[0] for layer in attn])
    indices = torch.arange(attn.shape[-1])
    attn[:, :, indices, indices] = 0
    
    if window_size==None:
        for (l,h) in headsets:
            entropy_vals =[]
            for w in range(attn.shape[2]):
                entropy_val = scipy.stats.entropy(attn[l][h][w],base=2)
                entropy_vals.append(entropy_val if entropy_val>0 else 0)
            entropies_all.append(entropy_vals)
    else:
        attn = attn[:,:,-sent_len:,:]
        for (l,h) in headsets:
            entropy_vals =[]
            for w in range(attn.shape[2]):
                entropy_val = scipy.stats.entropy(attn[l][h][w][:-(sent_len-w)][-window_size:],base=2)
                entropy_vals.append(entropy_val if entropy_val>0 else 0)
            entropies_all.append(entropy_vals)

    entropies_all_tensor = torch.tensor(entropies_all)
    global_entropy = torch.mean(entropies_all_tensor, dim=(0))
    global_entropy_rounded_floats = [round(float(val),4) for val in global_entropy]
        
    return global_entropy_rounded_floats



def find_token_position(l,t):
    return [i for i in range(len(l)) if l[i]==t]


def format_attention(attention):
   squeezed = []
   for layer_attention in attention:
      squeezed.append(layer_attention.squeeze(0))
   return torch.stack(squeezed)


def get_paid_attention(sent, source_token, target_token, head):
    indexed_tokens = tokenizer.encode_plus(text = sent, return_tensors = 'pt', add_special_tokens = False)['input_ids'][0]
    
    with torch.no_grad():
        attention = model(indexed_tokens)[-1]
        attention = format_attention(attention)

    tokens = tokenizer.convert_ids_to_tokens(indexed_tokens)
    tokens = [tokenizer.decode(i).strip() for i in indexed_tokens]
    source_token_positions, target_token_positions = find_token_position(tokens,source_token), find_token_position(tokens,target_token)
    
    if len(source_token_positions) != 1 or len(target_token_positions)!= 1 : return None # If a target token does not apper or appear multiple times, the returned surprisal is zero
    else: source_token_position, target_token_position = source_token_positions[0], target_token_positions[0]

    l, h = head[0], head[1]
    return round(float(attention[l,h,source_token_position,target_token_position]),5)