import pandas as pd
from ChessTokenizer import ChessTokenizer
from runner import create_dataloader
import torch 
from gpt_model import GPTModel
import pandas as pd

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

    # Get Games in movelist
    df = pd.read_csv("games.csv")
    df = df.loc[df['rated']==True]
    column = df["moves"]
    gamesList = column.tolist()

    games = gamesList[:SAMPLE_SIZE]

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
    GPT_CONFIG["vocab_size"] = len(all_words)

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

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    train_loader
    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG, OTHER_SETTINGS)

# torch.save(model, "model_complete.pth")
# model.eval()