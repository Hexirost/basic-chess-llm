import pandas as pd
from ChessTokenizer import ChessTokenizer
from runner import GPTDataset, create_dataloader
import torch 

SAMPLE_SIZE = 4

CHUNK_SIZE = 4

# Get Games in movelist
df = pd.read_csv("games.csv")
column = df["moves"]
gamesList = column.tolist()

sample = gamesList[:SAMPLE_SIZE] # Take only a sample size for testing

# Make each game a block
games = [move for move in sample if move.strip()]

# Tokenize
allTokens = set()
allTokens.update([token for game in games for token in game.split(" ")])
tokenizer = ChessTokenizer(allTokens, ["<|unk|>"])

vocab_size = len(allTokens) + 1 # Size of all tokens; 1 is from unk char
print(allTokens)
output_dim = 8 # how much info(weights in vector) is stored per token
dataloader = create_dataloader(games[0], tokenizer=tokenizer, batch_size=4, max_length=CHUNK_SIZE, stride=1, shuffle=False) # Note maybe change the stride later if overfitting or taking to long

torch.manual_seed(123) # hardcode random seed For testing
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # Create embedding layer(weights)

# Create positional weight layer
context_length = CHUNK_SIZE
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(CHUNK_SIZE))


# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# token_embeddings = token_embedding_layer(inputs)
# input_embeddings = token_embeddings + pos_embeddings
