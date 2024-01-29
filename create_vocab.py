import numpy as np
import time
from collections import Counter
import json


def tokenize_word(word, token_dicts):
    if not word:
        return []

    for token_length in token_dicts.keys():
        if len(word) < token_length:
            continue
        tokens = token_dicts[token_length]
        for token in tokens:
            if token in word:
                parts = word.split(token)
                result = []
                for part in parts[:-1]:
                    if len(part) == 1:  # Если часть состоит из одного символа, считаем его токеном
                        result.append(part)
                    elif len(part) > 1:  # Если часть длиннее, вызываем рекурсию
                        result.extend(tokenize_word(part, token_dicts))
                    result.append(token)  # Добавляем текущий токен между частями

                # Обрабатываем последнюю часть слова
                if len(parts[-1]) == 1:
                    result.append(parts[-1])
                elif len(parts[-1]) > 1:
                    result.extend(tokenize_word(parts[-1], token_dicts))

                return result


    return [word]

def char_pairs_entropy(char_pairs: dict):
    """Calculate entropy of char pairs
        - return float
    """
    dict_sum = sum(char_pairs.values())
    H = 0
    for pair, count in char_pairs.items():
        H += count/dict_sum * np.log2(count/dict_sum)
    return -H, dict_sum

#count pairs in tokens and return most frequent pair
def max_pair(tokens):
    """Receive list of tokens, pair them and return most frequent pair."""
    char_pairs = Counter()
    for token in tokens:
        char_pairs.update(''.join(pair) for pair in zip(token, token[1:]))
    
    return char_pairs.most_common(1)[0], char_pairs


def bpe(data, vocab, target_vocab_size, min_token_freq):

    vocab_len = sum(len(v) for v in vocab.values())
    

    i = 0
    start = time.time()
    while vocab_len < target_vocab_size:
        #sort vocabulary by token length
        vocab = dict(sorted(vocab.items(), reverse=True))

        tokens = [tokenize_word(word, vocab) for word in data]
        total_tokens = sum(len(token) for token in tokens)

        (pair, count), char_pais = max_pair(tokens)
        h,_ = char_pairs_entropy(char_pais)
        if count < min_token_freq:
            print(f'\nEntropy at step {i}: {h}. Vocabulary size: {vocab_len}. Tokens in corpus: {total_tokens}')
            print(f'Iteration done in {time.time()-start:.2f} sec')
            print("Minimum token frequency reached. Stopping...")
            break

        token_set = vocab.get(len(pair), set())
        token_set.add(pair)
        vocab[len(pair)] = token_set

        vocab_len = sum(len(v) for v in vocab.values())
   

        if i % 10 == 0:
            print(f'\nEntropy at step {i}: {h}. Vocabulary size: {vocab_len}. Tokens in corpus: {total_tokens}')
            print(f'Iteration done in {time.time()-start:.2f} sec')
            print("Continuing...", end = '', flush=True)
            start = time.time()
        else:
            print('.', end='', flush=True)
        #add new token to vocabulary
        i += 1 
    print(f"Vocabulary created with length {vocab_len}")
    
    #addinig special tokens to vocabulary
    token_set = vocab.get(8, set())
    token_set.add("D0U0D0V0")
    vocab[8] = token_set
    vocab = dict(sorted(vocab.items(), reverse=True))

    return vocab

def init_vocab(data):
    """Initialize vocabulary with all characters in the data."""
    vocab = {1: set(''.join(data))}
    

    return vocab

def main(args):
    

    #read data txt file
    with open(args.data_path, 'r') as f:
        data = f.read().splitlines()
    
    #initiate vocabulary with all characters in the data
    vocab = init_vocab(data)
    
    #cut start and end chars from words
    data_words = [line[1:-1] for line in data]

    #create vocabulary using bpe algorithm
    target_vocab_size = args.vocab_size
    min_token_freq = args.min_token_freq

    print(f'Creating vocabulary with up to {target_vocab_size} tokens')
    vocab = bpe(data_words, vocab, target_vocab_size, min_token_freq)

    for key in vocab.keys():
        vocab[key] = list(vocab[key])


    with open(f'{args.vocab_path}_{target_vocab_size+1}.json', 'w', encoding='utf-8') as file:
        json.dump(vocab, file)
        print('Vocabulary saved')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Create vocabulary using BPE algorithm")
    parser.add_argument("--data_path", type=str, default='data/NQ_moves_full_vol.txt', help="Path to text data file")
    parser.add_argument("--vocab_size", type=int, default=98, help="Target vocabulary size")
    parser.add_argument("--vocab_path", type=str, default='data/NQ_vocab_vol', help="Path to save vocabulary")
    parser.add_argument("--min_token_freq", type=int, default=20000, help="Minimum token frequency")
    args = parser.parse_args()

    main(args)