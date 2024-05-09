import torch
import torch.nn as nn
from Berryizer import Berryizer
from torch.nn import functional as F

# ---- Hyperparams
batch_size = 64
"""Number of parallel independent batches of data to get when training or evaluating"""
block_size = 256
"""Context length in number of tokens (number of tokens used to predict the next token)"""
vocab_size = 1500
"""Number of tokens known and usable"""
n_embed = 208
"""Number of dimensions for the embeddings"""
n_head = 4
"""Number of self-attention heads in the multi-head attention block"""
n_layer = 8
"""Number of consecutive layer of communication and computation blocks"""
dropout = 0.2
"""Probability for some input tensor's element to dropout on each forward call"""
max_iters = 7000
"""Number of iterations for the training loop"""
eval_interval = 500
"""Number of iterations between each evaluation in the training loop"""
eval_iters = 200
"""Number of iterations for the evaluation loop"""
learning_rate = 1e-2
"""The model learning rate"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""Use CUDA if available instead of the CPU to improve speed"""
context = "The One Piece is"
"""The starting prompt that should be completed by the transformer (better not to end it with a space)"""
# ----


torch.set_default_dtype(torch.bfloat16)                     # Set the default dtype to bfloat16 (instead of float32) for faster computations
torch.cuda.empty_cache()                                    # Make space in the GPU memory
with open("utils/OnePieceSummary.txt", 'r', -1, 'utf-8') as f:
    text = f.read()                                         # Get the OnePieceSummary text


# ---- Tokenizer
b = Berryizer()
if not b.load(f"utils/Berryizer_{vocab_size}.model"):       # Use the saved Berryizer's vocab if it exists
    b.train(text=text, vocab_size=vocab_size)               # Otherwise, train the Berryizer on the OP summary text and generate a vocab
    b.save(f"utils/Berryizer_{vocab_size}")                 # and save the generated Berryizer's vocab
berryized_text = b.encode(text)                             # Encode the text into a list of tokens
# ----


# Split the dataset into a train (95%) and eval (5%) tensors
data = torch.tensor(berryized_text, dtype=torch.long)
n = int(0.95 * len(data))
train_data = data[:n]                                       # The first 95% of the dataset
eval_data = data[n:]                                        # The remaining 5% of the dataset


def get_batch(split):
    """Get random batches of data from the @split dataset.

    split -- 'train' for the training dataset, anything else default to the evaluating dataset
    """
    data = train_data if split == 'train' else eval_data    # either get from the training or evaluating dataset
    ix = torch.randint(len(data)-block_size, (batch_size,)) # (indexes) generate @batch_size number of random index in the dataset
    x = torch.stack([data[i:i+block_size] for i in ix])     # (inputs) stack of @batch_size number of @block_size number of tokens starting from i
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (outputs) stack of @batch_size number of @block_size number of tokens starting from i+1
    x, y = x.to(device), y.to(device)                       # send those stacks to the defined device's memory
    return x, y


@torch.no_grad
def estimate_loss():
    """Estimate the mean loss over @eval_iters number of iterations for each of the training and evaluating datasets."""
    out = {}
    m.eval()                                                # set the model in eval mode
    for split in ['train', 'val']:                          # loop once for the training dataset and then once more for the evaluating dataset
        losses = torch.zeros(eval_iters)                    # generate a tensor of @eval_iters size filled with zeroes
        for k in range(eval_iters):                         # loop @eval_iters times
            X, Y = get_batch(split)                         # get batches from the current dataset
            logits, loss = m(X, Y)                          # get the loss from the model's forward call
            losses[k] = loss.item()                         # update the loss value for the current batch
        out[split] = losses.mean()                          # get the mean loss for the current dataset
    m.train()                                               # reset the model to train mode
    return out                                              # return the mean loss for each datasets


class Head(nn.Module):
    """Single self-attention head.
    
    A self-attention head basically try to figure-out for each tokens what preceding tokens attend to them the most. It enrich 
    the information of the token based on it's preceding context.
    
    An way to visualize it would be with the phrase "A small animal roamed the verdant forest". Here the token 'small' add context
    to the token 'animal', same for the token 'verdant' with 'forest'. However, the token 'small' add no direct context to
    the token 'forest'.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                                         # (B, T, C)
        q = self.query(x)                                                       # (B, T, C)
        # Normalized scaled attention scores (scale is 1/C^.5)
        # For each token, find which preceding tokens are the more contextually important and linked
        weights = q @ k.transpose(-2, -1) * C**-0.5                             # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        weights = F.softmax(weights, dim=-1)                                    # (B, T, T)
        weights = self.dropout(weights)                                         # (B, T, T)
        # Weighted values' aggregation
        v = self.value(x)                                                       # (B, T, C)
        return weights @ v                                                      # (B, T, T) @ (B, T, C) -> (B, T, C)


class MultiHeadAttention(nn.Module):
    """Parallel multiple heads of self-attention.
    
    Execute multiple self-attention heads in parallel, all of these attention scores are then combined together.
    This give the transformer the power to encode multiple relationships and add more nuances for each tokens.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A Transformer block: communication then computation"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x)) # Apply heads of self-attention  (B, T, C)
        x = x + self.ffwd(self.ln2(x))                                #    (B, T, C)
        return x


class OnePieceTransformer(nn.Module):
    """A basic transformer NLP model"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)          # Encode 'the meaning' of each token from the vocab in an embedding of @n_embed dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embed)       # Encode the position within a block in an embedding of @n_embed dimensions
        # Apply communication followed by computation multiple times
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape                                                        # Get the shape, or size, for each dimension of the @idx tensor (B x T)

        # idx and targets are (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)                               # Get the embedding for 'the meaning' of each token in the @idx sequence of tokens. -> Embedding(@batch_size, @block_size, @n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Get the position embedding for each integer value in the range (0, @T). -> Embedding(@block_size, @n_embed)
        x = tok_emb + pos_emb                                                   # Add the position embeddings to each corresponding token embedding. -> Embedding(@batch_size, @block_size, @n_embed)
        x = self.blocks(x)                                                      # Apply the sequence of communication and computation blocks. -> Embedding(@batch_size, @block_size, @n_embed)
        x = self.ln_f(x)                                                        # Apply the layer normalization. -> Embedding(@batch_size, @block_size, @n_embed)
        logits = self.lm_head(x)                                                # Apply the linear transformation. -> Embedding(@batch_size, @block_size, @n_embed)

        # If a @targets target sequence is given, calculate the loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Return the concatenation of the @max_new_tokens generated tokens and the input sequence of tokens (@idx)."""
        for _ in range(max_new_tokens):                                         # Loop @max_new_tokens times
            corr_idx = idx[:, -block_size:]                                     # Select a sequence of the last n∈[0, @block_size] tokens from the input sequence
            logits, loss = self(corr_idx)                                       # Get the logits for the next token based on this sequence
            logits = logits[:, -1, :]                                           # Only select the last column of the next token's logits
            probs = F.softmax(logits, dim=-1)                                   # Apply a softmax to get a probability distribution (adds up to 1)
            idx_next = torch.multinomial(probs, num_samples=1)                  # Get the index of the logit with the highest probability
            idx = torch.cat((idx, idx_next), dim=1)                             # Add this new index to the end of the sequence
        
        return idx                                                              # Return the initial sequence of tokens concatenated with @max_new_tokens new tokens

    def exercise(self):
        """Train the model"""
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = get_batch('train')

            logits, loss = self(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def save(self, path="utils/OnePieceTransformer.model"):
        """Save the model parameters to the given @path path"""
        torch.save(self.state_dict(), path)
        
    def load(self, path="utils/OnePieceTransformer.model"):
        """Load the model parameters from the given @path path"""
        try:
            self.load_state_dict(torch.load(path))
            return True
        except OSError:
            return False

model = OnePieceTransformer()
m = model.to(device)                                        # Send the model computations to GPU through CUDA if available

# PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Load or train and save the model
if not m.load():                                            # Try to load the model
    m.exercise()                                            # if fail, train the model
    m.save()                                                # and save the model

# Generate from the model
context = torch.tensor([b.encode(context)], dtype=torch.long, device=device)
print(b.decode(m.generate(context, block_size)[0].tolist()))