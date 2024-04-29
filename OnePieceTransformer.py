import torch
import torch.nn as nn
from Berryizer import Berryizer
from torch.nn import functional as F

# ---- Hyperparams
batch_size = 32      # n of parallel independent sequences
block_size = 256     # n of prediction's max context length (context window)
vocab_size = 1024
n_embed = 192
n_head = 6
n_layer = 6
dropout = 0.2
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 2e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
context = "\n\nLuffy questions Vegapunk about the One Piece treasure"
# ----

torch.cuda.empty_cache()
with open('OnePieceSummary.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# ---- Tokenizer
b = Berryizer()
if not b.load(f"Berryizer_{vocab_size}.model"): # Use the saved Berryizer's vocabulary if it exists
    b.train(text=text, vocab_size=vocab_size)   # Otherwise, Train the Berryizer on the OP summary text and generate a vocab
    b.save(f"Berryizer_{vocab_size}")
berryized_text = b.encode(text)
# ----


# Divide text into train and test parts
data = torch.tensor(berryized_text, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Load data
def get_batch(split):
    # select random batch of data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])     # inputs
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # targets
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Single self-attention head"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, C)
        q = self.query(x)   # (B, T, C)
        # Normalized scaled attention scores (scale is 1/C^.5)
        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1)                                 # (B, T, T)
        weights = self.dropout(weights)
        # Weighted values' aggregation
        v = self.value(x)   # (B, T, C)
        return weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)


class MultiHeadAttention(nn.Module):
    """Parallel multiple heads of self-attention"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

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


class BigramLangageModel(nn.Module):
    """Basique bigram language model"""
    def __init__(self):
        super().__init__()
        # Each tokens get the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # Apply communication followed by computation multiple times
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are (B, T) tensor of ints
        tok_emb = self.token_embedding_table(idx)                               # (batch_size, block_size, n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (block_size, n_embed)
        x = tok_emb + pos_emb                                                   # (B, T, C)
        x = self.blocks(x)                                                      # (B, T, C)
        x = self.ln_f(x)                                                        # (B, T, C)
        logits = self.lm_head(x)                                                # (batch_size, block_size, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            corr_idx = idx[:, -block_size:] # Make sure that idx doesn't overflow the embedding (max: block_size)
            logits, loss = self(corr_idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


model = BigramLangageModel()
m = model.to(device)

# PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.tensor([b.encode(context)], dtype=torch.long, device=device)
print(b.decode(m.generate(context, max_new_tokens=300)[0].tolist()))

# Exemple results:
# [...]
# step 4500: train loss 2.3755, val loss 3.0537 -> Overfitting, probably due to new concepts and words appearing in test set not present in train set
#
#
# Luffy questions Vegapunk about the One Piece treasure.
# Luffy left off the meat Pirates and discloses to Usopp's house. While the B Building, everyone cast some large door coward. He else sequire pirate allowed tied up. Kin'emon asked. A whipost gives into a room where the Marines are.
# On the bride, Nami, the mend is flocks Luffy and Zoro has a means this battle be in the return. Kin'emon starts to cry over the desert and Chopper. Unowned, the Dies of the World Government has been shocked by a Log Pose, originallyowever they were about to continue running. They then use Enel and follow him. the people with King Climous.
# Sear Buchierates of Skypiea impressed, saying that Luffy and Zoro want the key for Monster Pose in action and turns to celeber and she rete to the movements. He was relieved that the island since she might have been killed and promised Monkey D. Luffy and his time he saw that determined'em, to get his chan