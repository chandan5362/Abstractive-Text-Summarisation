
import re
from collections import defaultdict
import argparse
import pickle

# read the train
class BPE(object):
    """Return the BPE vocab dictionary with the tokens and their corresponding frequency"""

    def __init__(self, train_path, test_path, vocab_size=50000, min_freq=2):
        self.train_path = train_path
        self.test_path = test_path
        self.min_freq = min_freq
        self.vocab_size = vocab_size

    def create_char_vocab(self):
        # read the train
        vocab = defaultdict(int)
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for sent in f:
                words = sent.strip().split()
                for word in words:
                    vocab[' '.join(list(word)) + ' </w>'] += 1
        return vocab

    def get_pair_counts(self, vocab):
        pair_counts = defaultdict(int)
        for k, v in vocab.items():
            chars = k.split()
            for i in range(len(chars)-1):
                pair_counts[(chars[i], chars[i+1])]+=1
        return pair_counts  


    def merge_pairs(self,best_pair, vocab_in):
        """recusrsively merge the frequently occuring character pairs"""
        vocab_out = {}
        bigram = re.escape(" ".join(best_pair))

        p = re.compile(r'(?<!\S)'+bigram + r'(?!\S)')
        for word in vocab_in:
            w_out = p.sub(''.join(best_pair), word)
    
            vocab_out[w_out] = vocab_in[word]
        return vocab_out    

    def get_tokens_count(self, vocab):
        """return the total number of tokens in the vocab"""
        char_tokens = defaultdict(int)
        
        for word, count in vocab.items():
            tokens = word.split()
            for t in tokens:
                char_tokens[t]+=count
                
        return char_tokens  

    def create_bpe_vocab(self, iterations):
        vocab = self.create_char_vocab()
        """create the BPE vocab"""
        iters = iterations
        for _ in range(iters):
            pairs = self.get_pair_counts(vocab)
            if not pairs:
                print("No pairs found")
                break

            #get the best pair with the highest count
            best_pair = max(pairs, key = pairs.get)
            # print(best_pair)
            vocab = self.merge_pairs(best_pair, vocab)

        #sort the vocabulary with respct to the frequency of the tokens
        print(len(vocab))
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size]

        #filter the vocab based on the min_freq
        vocab = {k:v for k, v in vocab if v >= self.min_freq}


        # print(vocab)
        return vocab


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train_path", required=True, help="path to the train file")
    ap.add_argument("-e", "--test_path", required=True, help="path to the test file")
    ap.add_argument("-v", "--vocab_size", required=False, help="vocab size", default=None)
    ap.add_argument("-m", "--min_freq", required=False, help="min freq", type=int, default=2)
    ap.add_argument("-i","--iterations", required=False, help="number of iterations for botoom up merging", type =int, default=100)
    ap.add_argument("-o", "--output_path", required=True, help="path to save the output bpe vocab")
    
    args = ap.parse_args()
    bpe = BPE(args.train_path, args.test_path, args.vocab_size, args.min_freq)
    out = bpe.create_bpe_vocab(args.iterations)

    #save the output vocab
    try:
        with open(args.output_path, 'wb') as f:
            pickle.dump(out, f)
        print("BP vocab successfully saved")
    except:
        print("Output path does not exist")
    

    # print("create bpe vocab:\n", out)

    
