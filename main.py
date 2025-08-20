import pandas as pd
from ChessTokenizer import ChessTokenizer

SAMPLE_SIZE = 4

# Get Games in movelist
df = pd.read_csv("games.csv")
column = df["moves"]
gamesList = column.tolist()

snip = gamesList[:SAMPLE_SIZE] # Take only a sample size for testing

# Make each game a block
games = [item for item in snip if item.strip()]

# Tokenize
allTokens = set()
allTokens.update([token for game in games for token in game.split(" ")])

all_words = sorted(list(allTokens))
all_words.extend(["<|unk|>"]) # Add Extra token for unknown values
vocab_size = len(all_words)
vocab = {token:integer for integer,token in enumerate(all_words)}
# TODO: Consider adding beginning of game and end of game and who won extra special tokens [BOS], [EOS]

tokenizer = ChessTokenizer(vocab)
