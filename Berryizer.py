import regex as re

def get_counts(ids, counts=None):
    """
    Get the number of appearence of each consecutive pair of tokens
     
    {
        (tokens[i], tokens[i+1]): n_appearance, ...
    }
    """
    counts = {} if counts is None else counts
    # Iterate over consecutive pairs of tokens -> and increment the pair's count
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, pairid):
    """
    Replace every occurence of the given pair in 'ids' by 'pairid'
    """
    lenids = len(ids)
    newids = []
    i = 0
    # For every token in 'ids'
    while i < lenids:
        # If not the last token, and this token matchs pair's first token, and the next token matchs pair's second token
        # -> append 'pairid' to the 'newids' list and go 2 index ahead
        if i < lenids-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(pairid)
            i += 2
        # Else -> append this token to the 'newids' list and go to the next index
        else:
            newids.append(ids[i])
            i += 1
    return newids

class Berryizer():
    def __init__(self):
        self.merges = {}
        self.regpat = r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_regpat = re.compile(self.regpat)
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """
        Create a dictionnary that return the bytes value for each token
        """
        vocab = {token: bytes([token]) for token in range(256)}
        for (p0, p1), pairid in self.merges.items():
            vocab[pairid] = vocab[p0] + vocab[p1]
        return vocab

    def _chunkode(self, bytestring):
        """
        Encode a chunk of text
        """
        tokens = list(bytestring)
        while len(tokens) >= 2:
            counts = get_counts(tokens)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # Nothing to merge
            pairid = self.merges[pair]
            tokens = merge(tokens, pair, pairid)
        return tokens

    def save(self, file_prefix="Berryizer"):
        """
        Saves two files: file_prefix.vocab and file_prefix.model:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write(f"{self.regpat}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = token.decode('utf-8', errors='replace')
                # Find the children of this token, if any
                if idx in inverted_merges:
                    # If this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = self.vocab[idx0].decode('utf-8', errors='replace')
                    s1 = self.vocab[idx1].decode('utf-8', errors='replace')
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # Otherwise just print it (this should just be the first 256 tokens)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file="Berryizer.model"):
        """
        Inverse of save() but only for the model file
        """
        assert model_file.endswith(".model")
        
        # Read the model file
        merges = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # Read the pattern
            self.regpat = f.readline().strip()
            self.compiled_regpat = re.compile(self.regpat)
            # Read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size):
        """
        Train the Berryizer on the given text 'text' to construct a vocabulary of the given size 'vocab_size'
        """
        assert vocab_size >= 256
        n_merges = vocab_size - 256

        # Separate the text into chunks to prevent 'bad' cross-words pairing 
        chunks = re.findall(self.compiled_regpat, text)

        # Encode each chunk to utf-8 and create a list of { ch_id: (char1, char2, ...) }
        ids = [list(chunk.encode("utf-8")) for chunk in chunks] # Each element is a list of the utf-8 encoded chars constituing this chunk
            
        # Recursively merge top pairs of tokens to create new ones
        merges = {}
        vocab = {id: bytes([id]) for id in range(256)}
        for i in range(n_merges):
            counts = {}
            for ch_id in ids:
                get_counts(ch_id, counts)

            pair = max(counts, key=counts.get)
            pairid = 256 + i
            ids = [merge(ch_id, pair, pairid) for ch_id in ids]
            merges[pair] = pairid
            vocab[pairid] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        """
        Encode a text into a list of tokens by first breaking it into chunks (MOTT words)
        """
        chunks = re.findall(self.compiled_regpat, text)
        tokens = []
        for chunk in chunks:
            bytestring = chunk.encode('utf-8')
            chunk_tokens = self._chunkode(bytestring)
            tokens.extend(chunk_tokens)
        return tokens

    def decode(self, tokens):
        """
        Decode a list of tokens back into a text
        """
        chunks = []
        for id in tokens:
            if id in self.vocab:
                chunks.append(self.vocab[id])
            else:
                raise ValueError(f"Invalid token id: {id}")

        bytestring = b"".join(chunks)
        return bytestring.decode("utf-8", errors="replace")