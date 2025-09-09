import pandas as pd
from ChessTokenizer import ChessTokenizer
from runner import create_dataloader
import torch 
from gpt_model import GPTModel
import pandas as pd
from tools import train_model_simple
from chess_games_tokenizer import chessSetup

GPT_CONFIG = {
    "context_length": 52,  # Shortened context length (orig: 1024)
    "emb_dim": 128,         # Embedding dimension
    "n_heads": 8,          # Number of attention heads
    "n_layers": 4,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 18,
    "batch_size": 8,
    "weight_decay": 0.1,
    "stride" : 1,
    "shuffle" : False,
    "num_workers" :0
}

SAMPLE_SIZE = 20

def main(gpt_config, settings):

    torch.manual_seed(123)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    tokenizer, processed, GPT_CONFIG["vocab_size"] = chessSetup(SAMPLE_SIZE)
    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    movesSet = " ".join(processed)
    train_ratio = 0.90
    split_idx = int(train_ratio * len(movesSet))

    train_loader = create_dataloader(
        movesSet[:split_idx],
        tokenizer=tokenizer,
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=OTHER_SETTINGS["stride"],
        drop_last=False,
        shuffle=OTHER_SETTINGS["shuffle"],
        num_workers=OTHER_SETTINGS["num_workers"],
    )

    val_loader = create_dataloader(
        movesSet[split_idx:],
        tokenizer=tokenizer,
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=OTHER_SETTINGS["stride"],
        drop_last=False,
        shuffle=OTHER_SETTINGS["shuffle"],
        num_workers=OTHER_SETTINGS["num_workers"],
    )


    ##############################
    # Train model
    ##############################


    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=10, eval_iter=1,
        start_context="<|start|> d4 d5 c4 c6", tokenizer = tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG, OTHER_SETTINGS)

# torch.save(model, "model_complete.pth")
# model.eval()