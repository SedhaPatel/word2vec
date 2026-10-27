# Word2Vec Skip-Gram From Scratch: Complete Command Reference

This guide stores all setup, training, and evaluation commands executed for the from-scratch Skip-Gram Word2Vec implementation (`word2vec_skipgram.py`), based on [Mikolov et al. (2013)](https://arxiv.org/html/1301.3781v3) and [Levy & Goldberg (2014)](https://aclanthology.org/W14-1618.pdf).

## 1. Project Directory Structure

```text
word2vec_project/
├── word2vec_skipgram.py           # From-scratch Word2Vec Skip-Gram & Analogy Engine
├── data/
│   ├── text8                      # Cleaned English Wikipedia corpus (17,005,207 tokens)
│   └── questions-words.txt        # Google Semantic-Syntactic Analogy benchmark (19,544 questions)
└── models/
    ├── vectors_text8.txt          # Trained 200D Skip-Gram embeddings (window=8, epochs=10)
    └── vectors_text8_300d.txt     # Trained 300D Skip-Gram embeddings (window=10, epochs=15)
```

## 2. Environment Setup & Dataset Download Commands

Run these commands inside your project directory (`/d/word2vec_project`) to create the folder structure and download the standard `text8` training corpus and Google `questions-words.txt` evaluation benchmark:

```bash
mkdir -p data models
```

```bash
curl -L -o data/text8.zip http://mattmahoney.net/dc/text8.zip
```

```bash
unzip data/text8.zip -d data/
```

```bash
curl -L -o data/questions-words.txt https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt
```

*(On Windows PowerShell without `unzip`, run `Expand-Archive data/text8.zip -DestinationPath data/`.)*

## 3. Built-In Verification Demo (No External Files Required)

Runs the self-contained natural-sentence relational corpus benchmark comparing exact online `word2vec.c` updates (`batch_size=1`), mini-batch Negative Sampling (`batch_size=64`), and Hierarchical Softmax (`SG-HS`):

```bash
python word2vec_skipgram.py
```

## 4. Model 1 (`200D`, `window=8`, `epochs=10`) — Training & Analogy Evaluation

### 4.1 Train 200D Skip-Gram with Negative Sampling on Full `text8`

```bash
python word2vec_skipgram.py --train data/text8 --save-vectors models/vectors_text8.txt --size 200 --window 8 --negative 10 --sample 1e-4 --min-count 5 --epochs 10 --batch-size 128
```

### 4.2 Evaluate 200D Vectors with `3CosAdd` (Linear Offset)

```bash
python word2vec_skipgram.py --load-vectors models/vectors_text8.txt --questions data/questions-words.txt --method 3cosadd
```

### 4.3 Evaluate 200D Vectors with `3CosMul` (Multiplicative Scoring)

```bash
python word2vec_skipgram.py --load-vectors models/vectors_text8.txt --questions data/questions-words.txt --method 3cosmul
```

## 5. Model 2 (`300D`, `window=10`, `epochs=15`) — Training & Analogy Evaluation

### 5.1 Train 300D Skip-Gram with Negative Sampling on Full `text8`

```bash
python word2vec_skipgram.py --train data/text8 --save-vectors models/vectors_text8_300d.txt --size 300 --window 10 --negative 10 --sample 1e-4 --min-count 5 --epochs 15 --batch-size 128
```

### 5.2 Evaluate 300D Vectors with `3CosAdd` (Linear Offset)

```bash
python word2vec_skipgram.py --load-vectors models/vectors_text8_300d.txt --questions data/questions-words.txt --method 3cosadd
```

### 5.3 Evaluate 300D Vectors with `3CosMul` (Multiplicative Scoring)

```bash
python word2vec_skipgram.py --load-vectors models/vectors_text8_300d.txt --questions data/questions-words.txt --method 3cosmul
```

## 6. Optional Quick Sanity-Check Command (2 Million Tokens)

For a fast 1–2 minute test run on the first 2 million tokens of `text8`:

```bash
python word2vec_skipgram.py --train data/text8 --max-tokens 2000000 --questions data/questions-words.txt --save-vectors models/vectors_2m.txt --size 100 --window 5 --negative 5 --min-count 5 --epochs 5 --batch-size 128
```

## 7. CLI Parameter Summary

| CLI Flag | Default | Description |
| :--- | :--- | :--- |
| `--train` | `None` | Path to plain-text training corpus (e.g., `data/text8`) |
| `--load-vectors` | `None` | Path to saved Word2Vec vector file for evaluation without retraining |
| `--save-vectors` | `None` | Output path to save trained `syn0` vectors in `word2vec.c` format |
| `--questions` | `None` | Path to Google `questions-words.txt` analogy benchmark file |
| `--size` | `100` | Embedding dimensionality $D$ (`200` and `300` used in main runs) |
| `--window` | `5` | Maximum dynamic context window size $C$ (`8` and `10` used in main runs) |
| `--min-count` | `5` | Minimum corpus frequency to include a word in the vocabulary |
| `--negative` | `5` | Number of negative samples $K$ drawn from $P_n(w) \propto U(w)^{0.75}$ |
| `--hs` | `False` | Enable Hierarchical Softmax via Huffman binary tree |
| `--cbow` | `False` | Train Continuous Bag-of-Words (CBOW) instead of Skip-Gram |
| `--sample` | `1e-3` | Frequent-word subsampling threshold $t$ (`1e-4` used on `text8`) |
| `--epochs` | `5` | Number of training passes over the corpus (`10` and `15` used on `text8`) |
| `--batch-size` | `64` | `1` for exact online `word2vec.c` SGD; `>1` for vectorized NumPy batches |
| `--restrict-vocab` | `30000` | Candidate vocabulary cap matching `threshold = 30000` in `compute-accuracy.c` |
| `--method` | `3cosadd` | Analogy scoring rule: `3cosadd` or `3cosmul` |
