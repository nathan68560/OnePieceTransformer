# OnePieceTransformer
## Summary
- [Summary](#summary)
- [Presentation](#presentation)
    + [Introduction](#introduction)
    + [Exemples](#exemples)
    + [Observations](#observations)
- [How it work](#how-it-work)
    1. [The dataset](#1-the-dataset)
    2. [The tokenizer](#2-the-tokenizer)
    3. [The transformer](#3-the-transformer)

## Presentation
### Introduction
**OnePieceTransformer** is a Transformer model developed in Python and who's purpose is to generate fictional "[One Piece](https://en.wikipedia.org/wiki/One_Piece)" adventures. My personal goal with this project was to learn more about the inner working of Transformers, and specifically GPTs and _Natural Language Processing (NLP)_, as well as to gain some more experiences in Python developement.

The idea for this project came after visioning [3Blue1Brown](https://www.youtube.com/@3blue1brown)'s episodes [5](https://www.youtube.com/watch?v=wjZofJX0v4M) and [6](https://www.youtube.com/watch?v=eMlx5fFNoYc) from his [Neural networks serie](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) that explains very well what Transformers are and how they work. [This article](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34), also helped me a lot to understand each step of how Transformers work.

The base code for the transformer and the tokenizer is based on [Andrej Karpathy](https://github.com/karpathy)'s excellent tutorials "[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6117s)" and "[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=4661s)".

### Exemples
1.  - **Input:** "The One Piece is"
    - **Result:** "The One Piece is boat. Her mad for marine money Den Den Den Mushi, where the man Law told the Revious sences of the festival. Predding condition that Pekoms wanted them to continue to take them free if they confirmation, will be catching in two years facution charges. Mont-d'Or tells Caesar that they will not exit. Corpses are given key to the captain, thinking this, saying he is how a book to die, saying that not he is a bounty. Coset tells the man about the presence that E the sakes was Lunit. Then the person snipeach is seen, but is then kicked into the large encrance. Bepo orders his men to fire against a shopping many chariod and the first bartender. Usopp further used to sneak, reflects to Water 7, saying they need to negotiate, which they agree. The should be stopped by them, but the Straw Hats find unreshapp"
    ----
2.  - **Input:** "Luffy"
    - **Result:** "Luffy is shown chastises to the ground, Zoro, who draws his attention to Hiyori, and Moria used his subordinates, Hiriluk, dancing symbols Hasmond' sniper. With even a lack of food, Shotaru announces that their foe crew mates more than be nothing. As the zombies cheer him in front of Moria, Luffy orders his right and Robin to make it on some vice admirals. The Thousand Sunny is hiding out of enough to defeat Moria as the Usopp Pirates all try to run to the bridge run away. Usopp finally makes it look for Luffy while Franky tells the seas creature to Luffy, Usopp, and Chopper, after realizing that they have to beaten. Nami interroges, calling him, whom Luffy and Franky. Another was very confused that Sanji and Usopp are located, and had stolen a Roy-Tokoro. They wonder how they should have stolen the savior of their most importance, and noting that after three years"
    ----
3.  - **Input:** "The Straw Hats arrive on a new island and find Enel inside of a big ship"
    - **Result:** "The Straw Hats arrive on a new island and find Enel inside of a big ship. The scene then begins talking among the nunction of the battle with a large tower parad bubbles. Sanji then shows consciousness on the Shandia, revealing that the island called him and there is a single person from the sinistern visitority that overheard so they even have to rescue. The crew then escape with a group of people and kiss Nami demolcy aramague. Wol notes how the bentoing the island and it is so could be disecuted's straight. The citizens are searched from one of the transmission. One of them passes by saying that the sea is surrounding them from reaching, not impossible. After going, Sanji uses Nigawativete to create a instmare where it exists. This caused Franky to remove the Wind the Sea in explain that his Jape is his only father, as a child sun in the ocean."

### Observations
The model fails to generate well constructed, meaningfull sentences. The model also sometimes create made-up words. This is most probably due to the quality of the dataset used for training, which consist of the 1100 firsts One Piece chapters' long summaries. So since the dataset is an aggregation of summaries of a progressing story, the transformer lacks of "time to learn" most of the non-reccurent elements of One Piece. This is also observable from the over-fitting that happens while training, with the train loss decreasing faster than the val loss as the training progress.

## How it work
### 1. The dataset
In order for a Transformer to work, it need to first be trained on data. This data should contains knowledge that we wish for the Transformer to learn or extrapolate from. In the case of the **OnePieceTransformer**, this knowledge is the story of "One Piece".

The file "[_OnePieceSummaryScrapper.py_](/OnePieceSummaryScrapper.py)" is responsible for generating the textfile that will contain this data. In order to do that, the script iterate through the URLs of the first 1100 chapters on the [One Piece fandom wiki](https://onepiece.fandom.com/wiki) and scrap the summary's text content of each before concatenating them all and saving them to "[_utils/OnePieceSummary.txt_](utils/OnePieceSummary.txt)".

### 2. The tokenizer
Now, in order to feed this raw text to the Transformer, it first needs to be encoded into tokens. In **OnePieceTransformer**'s case, tokens are numerical representations of UTF-8 encoded character(s). The script that manage this is the **Berryizer** class from the "[_Berryizer.py_](/Berryizer.py)" file.

#### Training
Before being able to use the **Berryizer** to encode anything, it needs to be trained on the text dataset to generate a vocabulary of the desired length. To generate the best vocabulary for a specific text with a specific length, the **Berryizer** use a [byte-pair-encoding algorithm](https://en.wikipedia.org/wiki/Byte_pair_encoding) which successively merge the most frequent pair of adjacent tokens to mint a new one. Here's a breakdown of how this work:
1. The **Berryizer** generate a dictionary that map the index of the first 256 unicode characters to their UTF-8 byte representation.
2. The training text is broken down into chunks (~words) using a regular expression.
3. Each chunks is converted into a list of characters, and each character is replaced with their index in the dictionary from step 1.
4. The most frequent pair of adjacent indexes accross all chunks mint a new entry in the dictionary for this pair.

    _This step is repeated until the dictionary is of desired length._
5. The training is complete.

#### Encoding
For the encoding, the **Berryizer** use the vocabulary constructed in the training to replace each token (i.e. initially characters) by their index in the vocabulary dictionnary, recursively, until there are no more replacement to do.

Exemple:
- vocabulary = [a: a, b: b, c: c, d: d, X: aa, Y: ab, Z: XY]
- text = "aaabdaaabac"
- encoding:
    1. "aaabdaaabac" -> "XabdXabac" (aa -> X)
    2. "XabdXabac" -> "XYdXYac" (ab -> Y)
    3. "XYdXYac" -> "ZdZac" (XY -> Z)

#### Decoding
The decoding is the exact opposite process of the encoding.

Exemple:
- vocabulary = [a: a, b: b, c: c, d: d, X: aa, Y: ab, Z: XY]
- text = "ZdZac"
- encoding:
    1. "ZdZac" -> "XYdXYac" (Z -> XY)
    2. "XYdXYac" -> "XabdXabac" (Y -> ab)
    3. "XabdXabac" -> "aaabdaaabac" (X -> aa)

#### Specificity
The main change in the **Berryizer** compared to the standard tokenizer presented in [Andrej Karpathy](https://github.com/karpathy)'s tutorial is that during the encoding step the **Berryizer** start by adding a space before the first word of every line. This change is inspired by [SentencePiece](https://github.com/google/sentencepiece)'s add_dummy_prefix option and prevent the tokenizer from generating two different tokens for a same word just because one is prefixed with a space and not the other (e.g. "hello" and " hello"). This small change result in a more optimized vocabulary.

### 3. The Transformer
#### Architecture
- OnePieceTransformer
    1. Embeddings
        - Token embedding (1500x208)
        - Position embedding (256x208)
    2. Blocks in sequence (x8)
        1. Masked multi-head attention
            1. Self-attention heads in parallel (x4)
                - Query: Linear transformation (208x52)
                - Key: Linear transformation (208x52)
                - Value: Linear transformation (208x52)
            2. Linear transformation (208x208)
        2. Feed-forward network
            1. Linear transformation (208x832)
            2. ReLU activation function
            3. Linear transformation (832x208)
    3. Linear transformation (208x1500)
        
The model also use skip connections and layer normalizations for both multi-head attention blocks and feed-forward networks, as well as a dropout mechanism with a probability of 0.2.