
#Some parts of code is adapted from
#https://leimao.github.io/blog/Byte-Pair-Encoding

import re
import pickle
from collections import defaultdict

class Tokeniser:
    """Tokenises the sentences and returns a list of lists of generated tokens"""

    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.load_vocab()

    def get_tokens_count(self, vocab):
        """return the total number of tokens in the vocab
        
        Parameters
        ----------
        vocab : dict
            the vocab dictionary
        """

        char_tokens = defaultdict(int)
        
        for word, count in vocab.items():
            tokens = word.split()
            for t in tokens:
                char_tokens[t]+=count
                
        return char_tokens  

    def load_vocab(self):
        """Loads the vocab dictionary from the vocab file"""

        self.vocab = pickle.load(open(self.vocab_file, 'rb'))
        self.vocab_tokenization = {''.join(word.split()):word for word in self.vocab.keys()}
        self.token_freqs = self.get_tokens_count(self.vocab)
        self.sorted_tokens_tuple = sorted(self.token_freqs.items(), key=lambda item: (self.measure_token_length(item[0]), item[1]), reverse=True)
        self.sorted_tokens = [token for (token, freq) in self.sorted_tokens_tuple]

    def measure_token_length(self, token):
        """returns the length of the token"""

        if token[-4:] == '</w>':
            return len(token[:-4]) + 1
        else:
            return len(token)

    
    def tokenize_word(self, string, sorted_tokens, unknown_token='</u>'):
        """returns the tokenised word if the given word is not a key in the bpe vocab dictionary
        
        Parameters
        ----------  
        string : str
            the word to be tokenised
        sorted_tokens : list
            the list of tokens sorted by frequency
        unknown_token : str
            the token to be used for unknown words
        """
        
        if string == '':
            return []
        if sorted_tokens == []:
            return [unknown_token]

        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token.replace('.', '[.]'))

            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            if len(matched_positions) == 0:
                continue
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += self.tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += self.tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            break
        if len(string_tokens)==0:
            string_tokens.append(unknown_token)
        return string_tokens


    def tokenized_postprocessing(self, word):

        # if that key alreaexists, return itself.
        if word in self.vocab_tokenization:
    
            tokenised_str = self.vocab_tokenization[word].split()
        #otherwise tokennise the word and then return it
        else:
    
            tokenised_str = self.tokenize_word(string=word, sorted_tokens=self.sorted_tokens, unknown_token='</u>')
        return tokenised_str



def tokenise_sentences(tokeniser:object, sentence: list, vocab_op_path:str, unknown_token='</u>'):
    """returns the tokenised sentence
    
    Parameters
    ----------
    tokeniser : object
        the tokeniser object
    sentence : list
        the sentence to be tokenised
    vocab_op_path : str
        the path to save the  toekns vocab file
    unknown_token : str
        the token to be used for unknown words
    """

    # for sentence in sentences:
    tokenized_sent = []
    tokenized_sentence = []

    #tokens vocabulary
    vocab = {w :v for (w,v) in tokeniser.sorted_tokens_tuple}
    #save the tokens vocabulary
    try:
        with open(vocab_op_path, 'wb') as f:
            pickle.dump(vocab, f)
        print("BP vocab successfully saved")
    except:
        print("Output path does not exist")
    
    word_index = {w :i for i, (w,_) in enumerate(vocab.items())}
    #assign the last index to unknown word
    word_index[unknown_token] = len(word_index)

    for word in sentence.strip().split():
        word = word+'</w>'
        for w in tokeniser.tokenized_postprocessing(word):
            tokenized_sentence.append(word_index[w])
            tokenized_sent.append(w)
    print(tokenized_sentence)


if __name__ == '__main__':
    tokeniser = Tokeniser('data/indiana/bpe_vocab.pkl')
    save_path = "data/indiana/vocabulary.pkl"
    sentence="normal heart size. mild unfolding and atherosclerotic calcification of the aorta. no focal air space consolidation. no pneumothorax or pleural effusion. visualized bony structures are unremarkable in appearance."
    tokenise_sentences(tokeniser, sentence, save_path )