import numpy as np
import time
from collections import Counter
import json
import create_vocab as vc
import h5py



def main():

    #locad vocab from json file
    vocab_file = 'data/NQ_vocab_vol_99.json'
    data_file = 'data/NQ_moves_full_vol.txt'
    encoded_data_file = 'data/encoded_NQ_data_vol_100.hdf5'
    idx_to_tok_file = 'data/NQ_idx_to_tok_vol_100.json'
    tok_to_idx_file = 'data/NQ_tok_to_idx_vol_100.json'
    tok_freq_file = 'data/NQ_tok_freq_vol_100.json'

    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    
    #change vocab keys from str to int
    vocab = {int(k):v for k,v in vocab.items()}

    vocab = dict(sorted(vocab.items(), reverse=True))

    #read data txt file
    with open(data_file, 'r') as f:
        data = f.read().splitlines()

    #tokenize data
    print('Tokenizing data...')
    tokens = [vc.tokenize_word(word, vocab) for word in data]

    #write tokens to txt file
    print('Writing tokens to file...')
    with open('data/NQ_tokens_vol_100.txt', 'w') as f:
        for word in tokens:
            f.write(' '.join(word)+'\n')

    #get frequencies of tokens
    print('Counting token frequencies...')
    token_freq = Counter()
    for token in tokens:
        token_freq.update(token)

    #sort tokens by frequency from most to least frequent
    token_freq = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)

    #save token frequencies to json file
    print('Saving token frequencies to file...')
    with open(tok_freq_file, 'w') as f:
        json.dump(token_freq, f)


    #create token to index  and back mappings with initial index = 1
    print('Creating token to index mappings...')
    token_to_index = {token[0]:i+1 for i, token in enumerate(token_freq)}
    index_to_token = {i+1:token[0] for i, token in enumerate(token_freq)}

    #encode tokens
    print('Encoding tokens...')
    encoded_tokens = []
    for token in tokens:
        encoded_tokens.extend([token_to_index[t] for t in token])

    #conevert to numpy array and save in h5 file
    print('Saving encoded tokens to file...')
    encoded_tokens = np.array(encoded_tokens)
    with h5py.File(encoded_data_file, 'w') as f:
        f.create_dataset('data', data=encoded_tokens)


    #save mappings to json files
    print('Saving mappings to json files...')
    with open(idx_to_tok_file, 'w') as f:
        json.dump(index_to_token, f)

    with open(tok_to_idx_file, 'w') as f:
        json.dump(token_to_index, f)

    print(f'Maximum token index: {max(index_to_token.keys())}')
    print('Done.')
if __name__ == '__main__':
    main()