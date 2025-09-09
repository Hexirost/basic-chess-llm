import pandas as pd
from ChessTokenizer import ChessTokenizer

def chessSetup(num_of_game):
    
    # Get Games in movelist
    df = pd.read_csv("ChessData/games.csv")
    df = df.loc[df['rated']==True]
    column = df["moves"]
    gamesList = column.tolist()

    games = gamesList[:num_of_game]

    bookendGame = [(["<|start|>"]  + game.split(" ") + ["<|end|>"]) for game in games]
    processed =  []

    # Should I add padding to games before and after to make data "faster"
    for game in bookendGame:
        # num_moves = len(game)
        # if num_moves < 52:
        #     processed.extend(game + (["<|pad|>"]*(52-num_moves)))
        # else:
        processed.extend(game[:52])

    all_words = sorted(set(processed))
    all_words.extend(["<|unk|>"])
    tokenizer_vocab = {token:integer for integer,token in enumerate(all_words)}
    tokenizer = ChessTokenizer(tokenizer_vocab)
    return tokenizer, processed, len(all_words)