import re

"""
Simplified Tokenizer for chess games with allowed special chars
Takes in list of chess moves in a list
Can encode and decode tokens
"""
class ChessTokenizer:
    def __init__(self, tokenList, allowed_special = None):
        
        all_words = sorted(list(tokenList))
        if allowed_special:
            for value in allowed_special:
                all_words.extend([value]) # Add Extra token for unknown values

        vocab = {token:integer for integer,token in enumerate(all_words)}

        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        self.vocab_size = len(all_words)
    
    def encode(self, text):
        preprocessed = list(re.split(" ", text))
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed # Hardcoded unknown characters TODO: Fix Later
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return text