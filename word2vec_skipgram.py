#!/usr/bin/env python3
"""
Word2Vec Skip-Gram Implementation from Scratch (Mikolov et al., 2013)
=====================================================================

Faithful from-scratch implementation of the original Word2Vec papers and C code
(`word2vec.c` and `compute-accuracy.c`) using only Python standard libraries
and NumPy, tailored for semantic and syntactic word analogy tasks.

References:
1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).
   "Efficient Estimation of Word Representations in Vector Space." ICLR 2013.
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
   "Distributed Representations of Words and Phrases and their Compositionality."
   NIPS 2013.

Key Original Paper & `word2vec.c` Features Included:
- Vocabulary construction with `min_count` frequency pruning and sorting
- Frequent-word subsampling using the exact formula from Mikolov's `word2vec.c`
- Unigram table raised to the 3/4 power (`P(w)^0.75`) for O(1) negative sampling
- Huffman binary tree construction (`CreateBinaryTree`) for Hierarchical Softmax
- Precomputed sigmoid lookup table (`EXP_TABLE_SIZE = 1000`, `MAX_EXP = 6.0`)
  used directly during training with exact `|f| >= MAX_EXP` boundary handling
- Dynamic context window sampling (`b in [0, window - 1]`)
- Exact `neu1e` hidden-layer error accumulator matching `word2vec.c` online SGD
  (`batch_size=1`), plus an optional mini-batch mode (`batch_size > 1`) for speed
- Full `neu1e` context-word update via `np.add.at` and `alpha=0.05` default for CBOW
- Linear learning rate decay down to `starting_alpha * 0.0001`
- Vectorized analogy solver & benchmark evaluator (`compute-accuracy.c` 3CosAdd
  & 3CosMul) reporting both Seen-Question Accuracy and Strict All-Question Accuracy
- Corpus loader supporting both newline-delimited files and single-line corpora
  like `text8` via `MAX_SENTENCE_LENGTH = 1000` chunking (matching `word2vec.c`)
- Full CLI (`--train`, `--questions`, `--save-vectors`, `--load-vectors`)
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, field
import heapq
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# 1. Precomputed Sigmoid Table (faithful to `word2vec.c`)
# =============================================================================

EXP_TABLE_SIZE = 1000
MAX_EXP = 6.0
MAX_SENTENCE_LENGTH = 1000  # Matches `#define MAX_SENTENCE_LENGTH 1000` in `word2vec.c`


def build_exp_table(table_size: int = EXP_TABLE_SIZE, max_exp: float = MAX_EXP) -> np.ndarray:
    """
    Precomputes the sigmoid lookup table exactly as in Mikolov's `word2vec.c`:
        expTable[i] = exp((i / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        expTable[i] = expTable[i] / (expTable[i] + 1)
    """
    i = np.arange(table_size, dtype=np.float64)
    x = (i / float(table_size) * 2.0 - 1.0) * max_exp
    exp_x = np.exp(x)
    return (exp_x / (exp_x + 1.0)).astype(np.float32)


EXP_TABLE = build_exp_table()
EXP_TABLE_SCALE = EXP_TABLE_SIZE / (MAX_EXP * 2.0)


def lookup_sigmoid_ns(f: np.ndarray | float) -> np.ndarray | float:
    """
    Evaluates sigmoid for Negative Sampling exactly as in `word2vec.c`:
        if (f > MAX_EXP) return 1.0
        else if (f < -MAX_EXP) return 0.0
        else return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
    """
    if np.ndim(f) == 0:
        val = float(f)
        if val > MAX_EXP:
            return 1.0
        if val < -MAX_EXP:
            return 0.0
        idx = int((val + MAX_EXP) * EXP_TABLE_SCALE)
        if idx >= EXP_TABLE_SIZE:
            idx = EXP_TABLE_SIZE - 1
        return float(EXP_TABLE[idx])

    f_arr = np.asarray(f, dtype=np.float32)
    idx = np.clip(((f_arr + MAX_EXP) * EXP_TABLE_SCALE).astype(np.int32), 0, EXP_TABLE_SIZE - 1)
    out = EXP_TABLE[idx]
    out = np.where(f_arr > MAX_EXP, np.float32(1.0), out)
    out = np.where(f_arr < -MAX_EXP, np.float32(0.0), out)
    return out


def lookup_sigmoid_hs(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates sigmoid for Hierarchical Softmax exactly as in `word2vec.c`:
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    Returns `(sigmoid_values, valid_mask)` where `valid_mask` is 1.0 for
    `(-MAX_EXP < f < MAX_EXP)` and 0.0 outside (so out-of-range nodes are skipped).
    """
    in_bounds = (f > -MAX_EXP) & (f < MAX_EXP)
    idx = np.clip(((f + MAX_EXP) * EXP_TABLE_SCALE).astype(np.int32), 0, EXP_TABLE_SIZE - 1)
    return EXP_TABLE[idx], in_bounds.astype(np.float32)


# =============================================================================
# 2. Vocabulary & Huffman Tree Data Structures (`vocab_word` in `word2vec.c`)
# =============================================================================

@dataclass
class VocabWord:
    """Represents a single vocabulary entry (`struct vocab_word` in `word2vec.c`)."""
    word: str
    count: int
    index: int
    # Huffman tree path for Hierarchical Softmax:
    # `point` holds internal node indices from root to leaf; `code` holds 0/1 bits.
    code: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    point: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))


class Vocabulary:
    """
    Handles tokenization, frequency filtering (`min_count`), frequent-word
    subsampling, O(1) negative sampling unigram table (`P(w)^0.75`), and
    Huffman binary tree generation for Hierarchical Softmax.
    """

    def __init__(
        self,
        min_count: int = 2,
        sample: float = 1e-3,
        ns_exponent: float = 0.75,
        unigram_table_size: int = 1_000_000,
    ) -> None:
        self.min_count = min_count
        self.sample = sample
        self.ns_exponent = ns_exponent
        self.unigram_table_size = unigram_table_size

        self.entries: List[VocabWord] = []
        self.word2id: Dict[str, int] = {}
        self.id2word: List[str] = []
        self.counts: np.ndarray = np.empty(0, dtype=np.int64)
        self.total_words: int = 0
        self.keep_probs: np.ndarray = np.empty(0, dtype=np.float64)
        self.unigram_table: np.ndarray = np.empty(0, dtype=np.int32)

        # Padded Huffman tree matrices for vectorized Hierarchical Softmax
        self.hs_points: np.ndarray = np.empty((0, 0), dtype=np.int32)
        self.hs_targets: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.hs_mask: np.ndarray = np.empty((0, 0), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.entries)

    @staticmethod
    def tokenize(text: str, lowercase: bool = True) -> List[str]:
        """Standard alphanumeric word tokenizer."""
        if lowercase:
            text = text.lower()
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)

    @classmethod
    def load_corpus_file(
        cls,
        filepath: str,
        max_sentence_length: int = MAX_SENTENCE_LENGTH,
        max_tokens: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Loads a plain-text corpus file (such as `text8` or a multi-line text file)
        and splits lines into chunks of at most `max_sentence_length` (1,000 tokens),
        matching `MAX_SENTENCE_LENGTH` in `word2vec.c` so single-line corpora like
        `text8` (17M words on one line) are segmented identically to the C code.
        """
        sentences: List[List[str]] = []
        tokens_read = 0
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tokens = cls.tokenize(line)
                if not tokens:
                    continue
                if max_tokens is not None and tokens_read + len(tokens) > max_tokens:
                    tokens = tokens[: max_tokens - tokens_read]
                for i in range(0, len(tokens), max_sentence_length):
                    chunk = tokens[i : i + max_sentence_length]
                    if len(chunk) > 1:
                        sentences.append(chunk)
                tokens_read += len(tokens)
                if max_tokens is not None and tokens_read >= max_tokens:
                    break
        return sentences

    def build_vocab(
        self,
        corpus: Iterable[Sequence[str]],
        build_huffman: bool = False,
    ) -> None:
        """
        Builds vocabulary sorted by descending frequency (like `SortVocab` in `word2vec.c`),
        computes subsampling keep-probabilities, initializes the negative sampling
        unigram table, and optionally builds the Huffman tree.
        """
        raw_counts: collections.Counter[str] = collections.Counter()
        for sentence in corpus:
            raw_counts.update(sentence)

        filtered = [
            (word, count)
            for word, count in raw_counts.items()
            if count >= self.min_count
        ]
        filtered.sort(key=lambda item: (-item[1], item[0]))

        if not filtered:
            raise ValueError(
                f"Vocabulary is empty after applying min_count={self.min_count}. "
                "Lower min_count or provide a larger corpus."
            )

        self.entries = [
            VocabWord(word=w, count=c, index=idx)
            for idx, (w, c) in enumerate(filtered)
        ]
        self.word2id = {entry.word: entry.index for entry in self.entries}
        self.id2word = [entry.word for entry in self.entries]
        self.counts = np.array([entry.count for entry in self.entries], dtype=np.int64)
        self.total_words = int(self.counts.sum())

        self._compute_subsampling_probs()
        self._init_unigram_table()
        if build_huffman:
            self._create_binary_tree()

    def _compute_subsampling_probs(self) -> None:
        """
        Computes word retention probabilities using the exact formula from
        Mikolov's `word2vec.c`:
            ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1)
                  * (sample * train_words) / vocab[word].cn
        When `sample <= 0`, subsampling is disabled (keep_prob = 1.0).
        """
        if self.sample <= 0.0 or self.total_words == 0:
            self.keep_probs = np.ones(self.size, dtype=np.float64)
            return

        rel_freq = self.counts.astype(np.float64) / float(self.total_words)
        ratio = rel_freq / self.sample
        keep = (np.sqrt(ratio) + 1.0) * (1.0 / ratio)
        self.keep_probs = np.clip(keep, 0.0, 1.0)

    def _init_unigram_table(self) -> None:
        """
        Implements `InitUnigramTable()` from `word2vec.c`:
        Populates an integer lookup array where each word `w` occupies a slice
        proportional to `count(w)^0.75`, allowing O(1) negative sampling.
        """
        pow_counts = np.power(self.counts.astype(np.float64), self.ns_exponent)
        train_words_pow = float(pow_counts.sum())
        probs = pow_counts / train_words_pow

        cum_probs = np.cumsum(probs)
        slots = np.round(cum_probs * self.unigram_table_size).astype(np.int64)
        slots[-1] = self.unigram_table_size

        table = np.empty(self.unigram_table_size, dtype=np.int32)
        start = 0
        for idx, end in enumerate(slots):
            if end > start:
                table[start:end] = idx
                start = int(end)
        self.unigram_table = table

    def _create_binary_tree(self) -> None:
        """
        Implements `CreateBinaryTree()` from `word2vec.c`:
        Constructs a minimum-variance Huffman tree over the vocabulary so frequent
        words have short binary codes for Hierarchical Softmax.
        """
        vocab_size = self.size
        if vocab_size == 1:
            self.entries[0].code = np.array([0], dtype=np.int8)
            self.entries[0].point = np.array([0], dtype=np.int32)
            self.hs_points = np.zeros((1, 1), dtype=np.int32)
            self.hs_targets = np.ones((1, 1), dtype=np.float32)
            self.hs_mask = np.ones((1, 1), dtype=np.float32)
            return

        count = np.zeros(vocab_size * 2 + 1, dtype=np.int64)
        binary = np.zeros(vocab_size * 2 + 1, dtype=np.int8)
        parent_node = np.zeros(vocab_size * 2 + 1, dtype=np.int32)

        count[:vocab_size] = self.counts
        count[vocab_size:] = int(1e15)

        heap: List[Tuple[int, int]] = [
            (int(self.counts[i]), i) for i in range(vocab_size)
        ]
        heapq.heapify(heap)

        for a in range(vocab_size - 1):
            c1, min1i = heapq.heappop(heap)
            c2, min2i = heapq.heappop(heap)
            new_node = vocab_size + a
            count[new_node] = c1 + c2
            parent_node[min1i] = new_node
            parent_node[min2i] = new_node
            binary[min2i] = 1
            heapq.heappush(heap, (c1 + c2, new_node))

        root_node = 2 * vocab_size - 2
        max_len = 0
        for a in range(vocab_size):
            b = a
            code_bits: List[int] = []
            points: List[int] = []
            while b != root_node:
                code_bits.append(int(binary[b]))
                b = int(parent_node[b])
                points.append(b - vocab_size)

            self.entries[a].code = np.array(code_bits[::-1], dtype=np.int8)
            self.entries[a].point = np.array(points[::-1], dtype=np.int32)
            if len(points) > max_len:
                max_len = len(points)

        # Build padded arrays for vectorized mini-batch Hierarchical Softmax
        self.hs_points = np.zeros((vocab_size, max_len), dtype=np.int32)
        self.hs_targets = np.zeros((vocab_size, max_len), dtype=np.float32)
        self.hs_mask = np.zeros((vocab_size, max_len), dtype=np.float32)
        for a, entry in enumerate(self.entries):
            length = len(entry.point)
            if length > 0:
                self.hs_points[a, :length] = entry.point
                # In `word2vec.c`: g = (1 - vocab[word].code[d] - f) * alpha
                self.hs_targets[a, :length] = 1.0 - entry.code.astype(np.float32)
                self.hs_mask[a, :length] = 1.0


# =============================================================================
# 3. Skip-Gram Word2Vec Model (`TrainModelThread` in `word2vec.c`)
# =============================================================================

class Word2VecSkipGram:
    """
    From-scratch implementation of the Word2Vec Skip-Gram model (with optional
    CBOW support) following Mikolov et al. (2013) and `word2vec.c`.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        negative: int = 5,
        hs: bool = False,
        sample: float = 1e-3,
        alpha: Optional[float] = None,
        min_alpha: Optional[float] = None,
        epochs: int = 5,
        sg: bool = True,
        seed: int = 42,
    ) -> None:
        if negative <= 0 and not hs:
            raise ValueError("At least one of `negative > 0` or `hs=True` must be enabled.")

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.hs = hs
        self.sample = sample
        self.sg = sg
        # Match `word2vec.c` default alpha: 0.025 for Skip-Gram, 0.05 for CBOW
        self.alpha = alpha if alpha is not None else (0.025 if sg else 0.05)
        self.min_alpha = min_alpha if min_alpha is not None else self.alpha * 0.0001
        self.epochs = epochs
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.vocab = Vocabulary(min_count=min_count, sample=sample)

        # Model parameters matching `word2vec.c` naming:
        # syn0: input word embeddings (V x D)
        # syn1neg: output word embeddings for Negative Sampling (V x D)
        # syn1: internal Huffman tree node weights for Hierarchical Softmax (V x D)
        self.syn0: np.ndarray = np.empty((0, vector_size), dtype=np.float32)
        self.syn1neg: np.ndarray = np.empty((0, vector_size), dtype=np.float32)
        self.syn1: np.ndarray = np.empty((0, vector_size), dtype=np.float32)
        self._syn0_norm: Optional[np.ndarray] = None
        self.loss_history: List[float] = []

    def _init_net(self) -> None:
        """
        Implements `InitNet()` from `word2vec.c`:
        - `syn0` initialized uniformly in `[-0.5 / D, +0.5 / D]`
        - `syn1neg` and `syn1` initialized to zeros
        """
        v_size = self.vocab.size
        d = self.vector_size
        bound = 0.5 / float(d)
        self.syn0 = self.rng.uniform(-bound, bound, size=(v_size, d)).astype(np.float32)
        if self.negative > 0:
            self.syn1neg = np.zeros((v_size, d), dtype=np.float32)
        if self.hs:
            self.syn1 = np.zeros((v_size, d), dtype=np.float32)
        self._syn0_norm = None

    def train_pair_skipgram(
        self,
        input_word_id: int,
        target_word_id: int,
        lr: float,
    ) -> float:
        """
        Exact online single-pair Skip-Gram update mirroring `word2vec.c` (`batch_size=1`):
        - `input_word_id` (`last_word` in C): context word at window offset `c` whose
          embedding `syn0[input_word_id]` is used as the input vector `l1`.
        - `target_word_id` (`word` in C): center word at `sentence_position` whose
          Huffman path (`syn1`) and/or negative samples (`syn1neg`) are predicted.
        - Uses `EXP_TABLE` via `lookup_sigmoid_hs` / `lookup_sigmoid_ns` with exact
          `|f| >= MAX_EXP` handling.
        """
        u = self.syn0[input_word_id]  # shape (D,), `syn0[l1]` in C
        neu1e = np.zeros(self.vector_size, dtype=np.float32)
        pair_loss = 0.0

        # 1. Hierarchical Softmax (`if (hs)` block in `word2vec.c`)
        if self.hs:
            entry = self.vocab.entries[target_word_id]
            points = entry.point
            codes = entry.code
            if len(points) > 0:
                syn1_nodes = self.syn1[points]
                dots = syn1_nodes @ u
                # C skips nodes where f <= -MAX_EXP or f >= MAX_EXP
                sig, valid_mask = lookup_sigmoid_hs(dots)
                targets = 1.0 - codes.astype(np.float32)
                g = ((targets - sig) * lr * valid_mask).astype(np.float32)

                neu1e += g @ syn1_nodes
                self.syn1[points] += np.outer(g, u)

                probs = np.where(targets == 1.0, sig, 1.0 - sig)
                pair_loss -= float(np.sum(np.log(np.clip(probs, 1e-10, 1.0)) * valid_mask))

        # 2. Negative Sampling (`if (negative > 0)` block in `word2vec.c`)
        if self.negative > 0:
            rand_slots = self.rng.integers(0, self.vocab.unigram_table_size, size=self.negative)
            neg_ids = self.vocab.unigram_table[rand_slots]

            # Positive target update (d = 0, target = word, label = 1)
            v_pos = self.syn1neg[target_word_id]
            f_pos = float(np.dot(u, v_pos))
            sig_pos = float(lookup_sigmoid_ns(f_pos))
            g_pos = np.float32((1.0 - sig_pos) * lr)
            neu1e += g_pos * v_pos
            self.syn1neg[target_word_id] += g_pos * u
            pair_loss -= math.log(max(sig_pos, 1e-10))

            # Negative samples update (d = 1..negative, label = 0; skip if target == word)
            valid_negs = neg_ids[neg_ids != target_word_id]
            if len(valid_negs) > 0:
                v_neg = self.syn1neg[valid_negs]
                dots_neg = v_neg @ u
                sig_neg = lookup_sigmoid_ns(dots_neg)
                g_neg = ((-sig_neg) * lr).astype(np.float32)
                neu1e += g_neg @ v_neg
                np.add.at(self.syn1neg, valid_negs, np.outer(g_neg, u))
                pair_loss -= float(np.sum(np.log(np.clip(1.0 - sig_neg, 1e-10, 1.0))))

        self.syn0[input_word_id] += neu1e
        return pair_loss

    def _train_batch_skipgram(
        self,
        input_word_ids: np.ndarray,
        target_word_ids: np.ndarray,
        lr: float,
    ) -> float:
        """
        Vectorized mini-batch approximation of Skip-Gram for faster execution in
        Python/NumPy when `batch_size > 1`.
        """
        batch_size = len(input_word_ids)
        if batch_size == 0:
            return 0.0

        u = self.syn0[input_word_ids]  # (B, D)
        neu1e = np.zeros_like(u)       # (B, D)
        total_loss = 0.0

        # 1. Vectorized Hierarchical Softmax (`if (hs)`)
        if self.hs:
            pts = self.vocab.hs_points[target_word_ids]       # (B, M)
            targets = self.vocab.hs_targets[target_word_ids]  # (B, M)
            path_mask = self.vocab.hs_mask[target_word_ids]   # (B, M)

            syn1_nodes = self.syn1[pts]                       # (B, M, D)
            dots_hs = np.einsum("bd,bmd->bm", u, syn1_nodes)
            sig_hs, bound_mask = lookup_sigmoid_hs(dots_hs)
            mask = path_mask * bound_mask
            g_hs = ((targets - sig_hs) * lr * mask).astype(np.float32)

            neu1e += np.einsum("bm,bmd->bd", g_hs, syn1_nodes)

            # Normalize by per-node hit count in batch so the Huffman root node
            # does not experience a batch_size-x learning rate spike
            flat_pts = pts.ravel()
            active_weights = mask.ravel()
            node_counts = np.bincount(
                flat_pts, weights=active_weights, minlength=len(self.syn1)
            ).astype(np.float32)
            inv_node_counts = 1.0 / np.maximum(node_counts, 1.0)
            scaled_g_hs = g_hs * inv_node_counts[pts]

            np.add.at(
                self.syn1,
                flat_pts,
                (scaled_g_hs[:, :, None] * u[:, None, :]).reshape(-1, self.vector_size),
            )

            probs_hs = np.where(targets == 1.0, sig_hs, 1.0 - sig_hs)
            total_loss -= float(np.sum(np.log(np.clip(probs_hs, 1e-10, 1.0)) * mask))

        # 2. Vectorized Negative Sampling (`if (negative > 0)`)
        if self.negative > 0:
            v_pos = self.syn1neg[target_word_ids]  # (B, D)
            dots_pos = np.sum(u * v_pos, axis=1)
            sig_pos = lookup_sigmoid_ns(dots_pos)
            g_pos = ((1.0 - sig_pos) * lr).astype(np.float32)

            neu1e += g_pos[:, None] * v_pos

            rand_slots = self.rng.integers(
                0, self.vocab.unigram_table_size, size=(batch_size, self.negative)
            )
            neg_ids = self.vocab.unigram_table[rand_slots]  # (B, K)
            valid_mask = (neg_ids != target_word_ids[:, None]).astype(np.float32)

            v_neg = self.syn1neg[neg_ids]  # (B, K, D)
            dots_neg = np.einsum("bd,bkd->bk", u, v_neg)
            sig_neg = lookup_sigmoid_ns(dots_neg)
            g_neg = ((-sig_neg) * lr * valid_mask).astype(np.float32)

            neu1e += np.einsum("bk,bkd->bd", g_neg, v_neg)

            np.add.at(self.syn1neg, target_word_ids, g_pos[:, None] * u)
            np.add.at(
                self.syn1neg,
                neg_ids.ravel(),
                (g_neg[:, :, None] * u[:, None, :]).reshape(-1, self.vector_size),
            )

            loss_pos = -float(np.sum(np.log(np.clip(sig_pos, 1e-10, 1.0))))
            loss_neg = -float(np.sum(np.log(np.clip(1.0 - sig_neg, 1e-10, 1.0)) * valid_mask))
            total_loss += loss_pos + loss_neg

        np.add.at(self.syn0, input_word_ids, neu1e)
        return total_loss / float(batch_size)

    def _train_cbow_window(
        self,
        context_ids: List[int],
        target_id: int,
        lr: float,
    ) -> float:
        """
        Exact CBOW update matching `if (cbow)` in `word2vec.c`:
        1. Averages context vectors into `neu1` (`neu1[c] /= cw`).
        2. Updates `syn1` (HS) and/or `syn1neg` (NS) using `EXP_TABLE` and accumulates `neu1e`.
        3. Adds the full `neu1e` to every context word in the window via `np.add.at`.
        """
        if not context_ids:
            return 0.0

        neu1 = np.mean(self.syn0[context_ids], axis=0)
        neu1e = np.zeros(self.vector_size, dtype=np.float32)
        step_loss = 0.0

        if self.hs:
            entry = self.vocab.entries[target_id]
            points = entry.point
            codes = entry.code
            if len(points) > 0:
                syn1_nodes = self.syn1[points]
                dots = syn1_nodes @ neu1
                sig, valid_mask = lookup_sigmoid_hs(dots)
                targets = 1.0 - codes.astype(np.float32)
                g = ((targets - sig) * lr * valid_mask).astype(np.float32)
                neu1e += g @ syn1_nodes
                self.syn1[points] += np.outer(g, neu1)
                probs = np.where(targets == 1.0, sig, 1.0 - sig)
                step_loss -= float(np.sum(np.log(np.clip(probs, 1e-10, 1.0)) * valid_mask))

        if self.negative > 0:
            rand_slots = self.rng.integers(0, self.vocab.unigram_table_size, size=self.negative)
            neg_ids = self.vocab.unigram_table[rand_slots]
            valid_negs = neg_ids[neg_ids != target_id]
            all_ids = np.empty(1 + len(valid_negs), dtype=np.int32)
            all_ids[0] = target_id
            all_ids[1:] = valid_negs

            labels = np.zeros(len(all_ids), dtype=np.float32)
            labels[0] = 1.0

            v_out = self.syn1neg[all_ids]
            dots = v_out @ neu1
            sig = lookup_sigmoid_ns(dots)
            g = ((labels - sig) * lr).astype(np.float32)
            neu1e += g @ v_out
            np.add.at(self.syn1neg, all_ids, np.outer(g, neu1))
            probs = np.where(labels == 1.0, sig, 1.0 - sig)
            step_loss -= float(np.sum(np.log(np.clip(probs, 1e-10, 1.0))))

        np.add.at(self.syn0, np.asarray(context_ids, dtype=np.int32), neu1e)
        return step_loss

    def fit(
        self,
        sentences: Iterable[Sequence[str]],
        batch_size: int = 1,
        verbose: bool = True,
    ) -> "Word2VecSkipGram":
        """
        Builds vocabulary, initializes weights, and trains the Word2Vec model
        over `self.epochs` passes with dynamic context windows and linear
        learning rate decay.
        """
        sentence_list = list(sentences)
        self.vocab.build_vocab(sentence_list, build_huffman=self.hs)
        self._init_net()

        indexed_sentences: List[np.ndarray] = []
        for sent in sentence_list:
            ids = [self.vocab.word2id[w] for w in sent if w in self.vocab.word2id]
            if len(ids) > 1:
                indexed_sentences.append(np.array(ids, dtype=np.int32))

        total_expected_words = self.vocab.total_words * self.epochs
        words_processed = 0
        self.loss_history = []

        use_fast_batches = self.sg and (batch_size > 1)

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            epoch_pairs = 0

            order = self.rng.permutation(len(indexed_sentences))
            batch_inputs: List[int] = []
            batch_targets: List[int] = []

            for sent_idx in order:
                raw_sent = indexed_sentences[sent_idx]
                words_processed += len(raw_sent)

                progress = min(1.0, float(words_processed) / float(max(1, total_expected_words)))
                lr = max(self.min_alpha, self.alpha * (1.0 - progress))

                if self.sample > 0.0:
                    keep_p = self.vocab.keep_probs[raw_sent]
                    mask = self.rng.random(len(raw_sent)) < keep_p
                    sent = raw_sent[mask]
                    if len(sent) < 2:
                        continue
                else:
                    sent = raw_sent

                sent_len = len(sent)
                b_offsets = self.rng.integers(0, self.window, size=sent_len)

                for pos in range(sent_len):
                    target_word_id = int(sent[pos])  # `word` in `word2vec.c`
                    actual_win = self.window - int(b_offsets[pos])
                    start = max(0, pos - actual_win)
                    end = min(sent_len, pos + actual_win + 1)

                    if self.sg:
                        for ctx_pos in range(start, end):
                            if ctx_pos == pos:
                                continue
                            input_word_id = int(sent[ctx_pos])  # `last_word` in `word2vec.c`
                            if use_fast_batches:
                                batch_inputs.append(input_word_id)
                                batch_targets.append(target_word_id)
                                if len(batch_inputs) >= batch_size:
                                    b_loss = self._train_batch_skipgram(
                                        np.array(batch_inputs, dtype=np.int32),
                                        np.array(batch_targets, dtype=np.int32),
                                        lr=lr,
                                    )
                                    epoch_loss += b_loss * len(batch_inputs)
                                    epoch_pairs += len(batch_inputs)
                                    batch_inputs.clear()
                                    batch_targets.clear()
                            else:
                                epoch_loss += self.train_pair_skipgram(
                                    input_word_id=input_word_id,
                                    target_word_id=target_word_id,
                                    lr=lr,
                                )
                                epoch_pairs += 1
                    else:
                        ctx_ids = [
                            int(sent[ctx_pos])
                            for ctx_pos in range(start, end)
                            if ctx_pos != pos
                        ]
                        epoch_loss += self._train_cbow_window(ctx_ids, target_word_id, lr)
                        epoch_pairs += 1

            if use_fast_batches and batch_inputs:
                b_loss = self._train_batch_skipgram(
                    np.array(batch_inputs, dtype=np.int32),
                    np.array(batch_targets, dtype=np.int32),
                    lr=lr,
                )
                epoch_loss += b_loss * len(batch_inputs)
                epoch_pairs += len(batch_inputs)
                batch_inputs.clear()
                batch_targets.clear()

            avg_loss = epoch_loss / float(max(1, epoch_pairs))
            self.loss_history.append(avg_loss)
            if verbose and (epoch == 1 or epoch % max(1, self.epochs // 5) == 0 or epoch == self.epochs):
                mode_str = "Skip-Gram" if self.sg else "CBOW"
                obj_str = "+".join(
                    ([f"NS({self.negative})"] if self.negative > 0 else [])
                    + (["HS"] if self.hs else [])
                )
                batch_str = "online (batch=1)" if not use_fast_batches else f"batch={batch_size}"
                print(
                    f"[Epoch {epoch:02d}/{self.epochs:02d}] "
                    f"Mode={mode_str} ({obj_str}, {batch_str}) | "
                    f"Avg Loss={avg_loss:.4f} | LR={lr:.5f}"
                )

        self._normalize_vectors()
        return self

    # =========================================================================
    # 4. Vector Normalization, Similarity & Analogy Engine (`compute-accuracy.c`)
    # =========================================================================

    def _normalize_vectors(self) -> np.ndarray:
        """
        Computes L2-normalized embeddings (`M[a] /= len` in `compute-accuracy.c`).
        """
        norms = np.linalg.norm(self.syn0, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self._syn0_norm = (self.syn0 / norms).astype(np.float32)
        return self._syn0_norm

    @property
    def normalized_vectors(self) -> np.ndarray:
        if self._syn0_norm is None:
            return self._normalize_vectors()
        return self._syn0_norm

    def get_vector(self, word: str, normalize: bool = False) -> np.ndarray:
        """Returns the embedding vector for `word`."""
        w = word.lower()
        if w not in self.vocab.word2id:
            raise KeyError(f"Word '{word}' not in vocabulary.")
        idx = self.vocab.word2id[w]
        return self.normalized_vectors[idx] if normalize else self.syn0[idx]

    def most_similar(
        self,
        word: str,
        top_k: int = 5,
        restrict_vocab: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Finds the `top_k` closest words by cosine similarity (`distance.c` in Word2Vec).
        """
        w = word.lower()
        if w not in self.vocab.word2id:
            raise KeyError(f"Word '{word}' not in vocabulary.")

        w_idx = self.vocab.word2id[w]
        limit = min(self.vocab.size, restrict_vocab) if restrict_vocab else self.vocab.size
        norm_vecs = self.normalized_vectors[:limit]
        sims = norm_vecs @ self.normalized_vectors[w_idx]
        if w_idx < limit:
            sims[w_idx] = -np.inf  # Exclude query word itself

        top_indices = np.argsort(-sims)[:top_k]
        return [(self.vocab.id2word[int(i)], float(sims[int(i)])) for i in top_indices]

    def solve_analogy(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 5,
        method: str = "3cosadd",
        restrict_vocab: Optional[int] = 30000,
        eps: float = 1e-4,
    ) -> List[Tuple[str, float]]:
        """
        Solves the word analogy `a : b :: c : ?` (e.g., `man : king :: woman : ?`).

        Parameters
        ----------
        restrict_vocab : Optional[int]
            Matches `threshold = 30000` in Mikolov's `compute-accuracy.c`, restricting
            candidate vocabulary words to the top `restrict_vocab` most frequent words.
            Pass `0` or `None` to search the entire vocabulary.
        """
        a_l, b_l, c_l = a.lower(), b.lower(), c.lower()
        for token in (a_l, b_l, c_l):
            if token not in self.vocab.word2id:
                raise KeyError(f"Analogy query word '{token}' is out of vocabulary.")

        limit = min(self.vocab.size, restrict_vocab) if restrict_vocab else self.vocab.size
        ia, ib, ic = (
            self.vocab.word2id[a_l],
            self.vocab.word2id[b_l],
            self.vocab.word2id[c_l],
        )
        all_norm = self.normalized_vectors
        cand_norm = all_norm[:limit]

        if method.lower() == "3cosadd":
            target_vec = all_norm[ib] - all_norm[ia] + all_norm[ic]
            target_norm = float(np.linalg.norm(target_vec))
            if target_norm > 0.0:
                target_vec = target_vec / target_norm
            scores = cand_norm @ target_vec
        elif method.lower() == "3cosmul":
            sim_a = (cand_norm @ all_norm[ia] + 1.0) * 0.5
            sim_b = (cand_norm @ all_norm[ib] + 1.0) * 0.5
            sim_c = (cand_norm @ all_norm[ic] + 1.0) * 0.5
            scores = (sim_b * sim_c) / (sim_a + eps)
        else:
            raise ValueError(f"Unknown analogy method '{method}'. Choose '3cosadd' or '3cosmul'.")

        # Exclude input query words {a, b, c} (`compute-accuracy.c`)
        for q_idx in (ia, ib, ic):
            if q_idx < limit:
                scores[q_idx] = -np.inf

        k = min(top_k, limit)
        if k < limit:
            part_idx = np.argpartition(-scores, k - 1)[:k]
            top_indices = part_idx[np.argsort(-scores[part_idx])]
        else:
            top_indices = np.argsort(-scores)[:k]

        return [(self.vocab.id2word[int(i)], float(scores[int(i)])) for i in top_indices]

    def evaluate_analogies(
        self,
        analogies: Sequence[Tuple[str, str, str, str]],
        method: str = "3cosadd",
        top_k: int = 5,
        restrict_vocab: Optional[int] = 30000,
        eps: float = 1e-4,
    ) -> Dict[str, float | int | List[Dict[str, object]]]:
        """
        Vectorized batch evaluation of 4-tuples `(a, b, c, expected_d)` following
        Mikolov's `compute-accuracy.c` benchmark protocol (including `restrict_vocab`,
        defaulting to the top 30,000 most frequent words).
        """
        limit = min(self.vocab.size, restrict_vocab) if restrict_vocab else self.vocab.size
        valid_quads: List[Tuple[str, str, str, str, int, int, int, int]] = []

        for a, b, c, d in analogies:
            a_l, b_l, c_l, d_l = a.lower(), b.lower(), c.lower(), d.lower()
            if any(w not in self.vocab.word2id for w in (a_l, b_l, c_l, d_l)):
                continue
            ia, ib, ic, id_ = (
                self.vocab.word2id[a_l],
                self.vocab.word2id[b_l],
                self.vocab.word2id[c_l],
                self.vocab.word2id[d_l],
            )
            # In `compute-accuracy.c`, words outside the top `threshold` (30k) are skipped
            if max(ia, ib, ic, id_) >= limit:
                continue
            valid_quads.append((a_l, b_l, c_l, d_l, ia, ib, ic, id_))

        seen = len(valid_quads)
        total_q = len(analogies)
        if seen == 0:
            return {
                "total": total_q,
                "seen": 0,
                "correct_top1": 0,
                "correct_topk": 0,
                "top1_accuracy": 0.0,
                "topk_accuracy": 0.0,
                "strict_top1_accuracy": 0.0,
                "mrr": 0.0,
                "details": [],
            }

        cand_norm = self.normalized_vectors[:limit]  # (V_lim, D)
        ia_arr = np.array([q[4] for q in valid_quads], dtype=np.int32)
        ib_arr = np.array([q[5] for q in valid_quads], dtype=np.int32)
        ic_arr = np.array([q[6] for q in valid_quads], dtype=np.int32)
        id_arr = np.array([q[7] for q in valid_quads], dtype=np.int32)

        if method.lower() == "3cosadd":
            targets = cand_norm[ib_arr] - cand_norm[ia_arr] + cand_norm[ic_arr]  # (Q, D)
            t_norms = np.linalg.norm(targets, axis=1, keepdims=True)
            t_norms = np.where(t_norms == 0.0, 1.0, t_norms)
            targets = targets / t_norms
            scores = targets @ cand_norm.T  # (Q, V_lim)
        elif method.lower() == "3cosmul":
            sim_a = (cand_norm[ia_arr] @ cand_norm.T + 1.0) * 0.5
            sim_b = (cand_norm[ib_arr] @ cand_norm.T + 1.0) * 0.5
            sim_c = (cand_norm[ic_arr] @ cand_norm.T + 1.0) * 0.5
            scores = (sim_b * sim_c) / (sim_a + eps)
        else:
            raise ValueError(f"Unknown analogy method '{method}'. Choose '3cosadd' or '3cosmul'.")

        # Mask out input words {a, b, c} for each query row
        q_rows = np.arange(seen, dtype=np.int32)
        scores[q_rows, ia_arr] = -np.inf
        scores[q_rows, ib_arr] = -np.inf
        scores[q_rows, ic_arr] = -np.inf

        # Compute exact rank of expected_d via comparison (no full argsort needed)
        expected_scores = scores[q_rows, id_arr][:, None]
        ranks = 1 + np.sum(scores > expected_scores, axis=1)

        correct_top1 = int(np.sum(ranks == 1))
        correct_topk = int(np.sum(ranks <= top_k))
        mrr = float(np.mean(1.0 / ranks.astype(np.float64)))

        k = min(top_k, limit)
        details: List[Dict[str, object]] = []
        for idx, (a_l, b_l, c_l, d_l, _, _, _, _) in enumerate(valid_quads):
            row_scores = scores[idx]
            if k < limit:
                part = np.argpartition(-row_scores, k - 1)[:k]
                top_idx = part[np.argsort(-row_scores[part])]
            else:
                top_idx = np.argsort(-row_scores)[:k]

            top_preds = [(self.vocab.id2word[int(i)], round(float(row_scores[int(i)]), 4)) for i in top_idx]
            rank_val = int(ranks[idx])
            details.append(
                {
                    "query": f"{a_l} : {b_l} :: {c_l} : ?",
                    "expected": d_l,
                    "predicted_top1": top_preds[0][0],
                    "top_k_predictions": top_preds,
                    "rank": rank_val,
                    "correct_top1": rank_val == 1,
                }
            )

        return {
            "total": total_q,
            "seen": seen,
            "correct_top1": correct_top1,
            "correct_topk": correct_topk,
            "top1_accuracy": correct_top1 / float(seen),
            "topk_accuracy": correct_topk / float(seen),
            "strict_top1_accuracy": correct_top1 / float(max(1, total_q)),
            "mrr": mrr,
            "details": details,
        }

    @staticmethod
    def parse_google_analogy_file(filepath: str) -> Dict[str, List[Tuple[str, str, str, str]]]:
        """
        Parses the standard Google Word2Vec `questions-words.txt` analogy benchmark file.
        """
        sections: Dict[str, List[Tuple[str, str, str, str]]] = collections.defaultdict(list)
        current_section = "default"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(":"):
                    current_section = line[1:].strip()
                else:
                    parts = line.split()
                    if len(parts) == 4:
                        sections[current_section].append((parts[0], parts[1], parts[2], parts[3]))
        return dict(sections)

    def evaluate_google_analogy_file(
        self,
        filepath: str,
        method: str = "3cosadd",
        top_k: int = 5,
        restrict_vocab: Optional[int] = 30000,
    ) -> Dict[str, object]:
        """
        Evaluates a full `questions-words.txt` file and reports:
        1. Relative Accuracy on Seen Questions (matching Mikolov's `compute-accuracy.c`).
        2. Strict Unconditional Accuracy over All 19,544 Questions (treating OOV /
           unseen questions as automatic failures).
        """
        sections = self.parse_google_analogy_file(filepath)
        semantic_sections = {
            "capital-common-countries",
            "capital-world",
            "currency",
            "city-in-state",
            "family",
        }

        sem_total = sem_seen = sem_top1 = sem_topk = 0
        syn_total = syn_seen = syn_top1 = syn_topk = 0
        total_questions = 0
        mrr_weighted_sum = 0.0

        eff_restrict = restrict_vocab if (restrict_vocab and restrict_vocab > 0) else None

        print("\n" + "=" * 78)
        print(f"BENCHMARK EVALUATION: {filepath} (method={method}, restrict_vocab={eff_restrict})")
        print("=" * 78)

        for sec_name, quads in sections.items():
            q_count = len(quads)
            total_questions += q_count
            res = self.evaluate_analogies(
                quads, method=method, top_k=top_k, restrict_vocab=eff_restrict
            )
            seen = int(res["seen"])
            c1 = int(res["correct_top1"])
            ck = int(res["correct_topk"])
            acc1 = float(res["top1_accuracy"]) * 100.0
            acck = float(res["topk_accuracy"]) * 100.0
            mrr = float(res["mrr"])
            mrr_weighted_sum += mrr * seen

            is_syn = sec_name.startswith("gram") or (sec_name not in semantic_sections)
            if not is_syn:
                sem_total += q_count
                sem_seen += seen
                sem_top1 += c1
                sem_topk += ck
            else:
                syn_total += q_count
                syn_seen += seen
                syn_top1 += c1
                syn_topk += ck

            print(
                f"  {sec_name:28s}: Top-1 = {acc1:6.2f}% ({c1:4d}/{seen:4d}) | "
                f"Top-{top_k} = {acck:6.2f}% | MRR = {mrr:.3f}"
            )

        tot_seen = sem_seen + syn_seen
        sem_acc_seen = (sem_top1 / float(sem_seen) * 100.0) if sem_seen else 0.0
        syn_acc_seen = (syn_top1 / float(syn_seen) * 100.0) if syn_seen else 0.0
        tot_acc1_seen = ((sem_top1 + syn_top1) / float(tot_seen) * 100.0) if tot_seen else 0.0
        tot_acck_seen = ((sem_topk + syn_topk) / float(tot_seen) * 100.0) if tot_seen else 0.0
        tot_mrr = (mrr_weighted_sum / float(tot_seen)) if tot_seen else 0.0

        sem_acc_all = (sem_top1 / float(sem_total) * 100.0) if sem_total else 0.0
        syn_acc_all = (syn_top1 / float(syn_total) * 100.0) if syn_total else 0.0
        tot_acc1_all = ((sem_top1 + syn_top1) / float(total_questions) * 100.0) if total_questions else 0.0

        print("-" * 78)
        print("  [A] Relative Accuracy on Seen Questions (compute-accuracy.c protocol):")
        print(f"      Semantic Accuracy (Seen) : {sem_acc_seen:6.2f}% ({sem_top1}/{sem_seen})")
        print(f"      Syntactic Accuracy (Seen): {syn_acc_seen:6.2f}% ({syn_top1}/{syn_seen})")
        print(
            f"      Total Accuracy (Top-1)   : {tot_acc1_seen:6.2f}% ({sem_top1 + syn_top1}/{tot_seen}, "
            f"coverage={tot_seen}/{total_questions})"
        )
        print(f"      Total Accuracy (Top-{top_k})   : {tot_acck_seen:6.2f}% | Overall MRR = {tot_mrr:.3f}")
        print("-" * 78)
        print("  [B] Strict Unconditional Accuracy over ALL Questions (OOV counted as 0):")
        print(f"      Semantic Accuracy (All)  : {sem_acc_all:6.2f}% ({sem_top1}/{sem_total})")
        print(f"      Syntactic Accuracy (All) : {syn_acc_all:6.2f}% ({syn_top1}/{syn_total})")
        print(f"      Total Accuracy (All)     : {tot_acc1_all:6.2f}% ({sem_top1 + syn_top1}/{total_questions})")
        print("=" * 78)

        return {
            "semantic_accuracy": sem_acc_seen,
            "syntactic_accuracy": syn_acc_seen,
            "total_top1_accuracy": tot_acc1_seen,
            "total_topk_accuracy": tot_acck_seen,
            "strict_semantic_accuracy": sem_acc_all,
            "strict_syntactic_accuracy": syn_acc_all,
            "strict_total_top1_accuracy": tot_acc1_all,
            "overall_mrr": tot_mrr,
            "seen": tot_seen,
            "total": total_questions,
        }

    # =========================================================================
    # 5. Original `word2vec.c` Model Export & Import
    # =========================================================================

    def save_word2vec_format(
        self,
        filepath: str,
        binary: bool = False,
        vocab_filepath: Optional[str] = None,
    ) -> None:
        """
        Saves word vectors in the original `word2vec.c` format:
        Header line: `<vocab_size> <vector_size>\n`
        Followed by `<word> <float1> ... <floatD>\n` (or raw float32 bytes if binary=True).
        Optionally saves `<word> <count>\n` to `vocab_filepath` (`-save-vocab` in `word2vec.c`).
        """
        with open(filepath, "wb") as f:
            header = f"{self.vocab.size} {self.vector_size}\n".encode("utf-8")
            f.write(header)
            for idx, word in enumerate(self.vocab.id2word):
                vec = self.syn0[idx]
                if binary:
                    f.write(word.encode("utf-8") + b" " + vec.astype(np.float32).tobytes() + b"\n")
                else:
                    vec_str = " ".join(f"{val:.6f}" for val in vec)
                    f.write(f"{word} {vec_str}\n".encode("utf-8"))

        if vocab_filepath is not None:
            with open(vocab_filepath, "w", encoding="utf-8") as vf:
                for entry in self.vocab.entries:
                    vf.write(f"{entry.word} {entry.count}\n")

    @classmethod
    def load_word2vec_format(
        cls,
        filepath: str,
        binary: bool = False,
        vocab_filepath: Optional[str] = None,
    ) -> "Word2VecSkipGram":
        """
        Loads embeddings saved in original `word2vec.c` text or binary format.
        Note: Because `word2vec.c` vector files store only `<word> <floats>` (and
        not `syn1`/`syn1neg` or word counts), loaded models are for inference/analogy
        evaluation unless `vocab_filepath` (`-read-vocab`) is also supplied.
        """
        saved_counts: Dict[str, int] = {}
        if vocab_filepath is not None:
            with open(vocab_filepath, "r", encoding="utf-8") as vf:
                for line in vf:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        saved_counts[parts[0]] = int(parts[1])

        with open(filepath, "rb") as f:
            header = f.readline().decode("utf-8").strip().split()
            vocab_size, vector_size = int(header[0]), int(header[1])

            model = cls(vector_size=vector_size)
            model.syn0 = np.zeros((vocab_size, vector_size), dtype=np.float32)
            entries: List[VocabWord] = []

            for idx in range(vocab_size):
                if binary:
                    word_bytes = bytearray()
                    while True:
                        ch = f.read(1)
                        if ch == b" " or not ch:
                            break
                        if ch != b"\n":
                            word_bytes.extend(ch)
                    word = word_bytes.decode("utf-8")
                    raw_floats = f.read(4 * vector_size)
                    vec = np.frombuffer(raw_floats, dtype=np.float32).copy()
                else:
                    parts = f.readline().decode("utf-8").strip().split()
                    word = parts[0]
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)

                model.syn0[idx] = vec
                cnt = saved_counts.get(word, vocab_size - idx)
                entries.append(VocabWord(word=word, count=cnt, index=idx))

            model.vocab.entries = entries
            model.vocab.id2word = [e.word for e in entries]
            model.vocab.word2id = {e.word: e.index for e in entries}
            model.vocab.counts = np.array([e.count for e in entries], dtype=np.int64)
            model.vocab.total_words = int(model.vocab.counts.sum())
            model._normalize_vectors()
            return model


# =============================================================================
# 6. Natural Sentence Analogy Corpus & Online vs. Mini-Batch Verification Demo
# =============================================================================

def generate_analogy_training_corpus(repeats: int = 14, seed: int = 42) -> List[List[str]]:
    """
    Generates a natural-sentence relational corpus covering four classic
    Mikolov et al. (2013) semantic and syntactic analogy categories:
    1. Royalty & Gender (`man : king :: woman : queen`, `father : mother :: brother : sister`)
    2. Capital Cities & Countries (`france : paris :: japan : tokyo`)
    3. Comparative & Superlative Adjectives (`fast : faster :: tall : taller`)
    4. Verb Tense Inflections (`walk : walked :: swim : swam`)
    """
    rng = np.random.default_rng(seed)
    templates: List[str] = []

    # -------------------------------------------------------------------------
    # Domain 1: Gender & Royalty
    # -------------------------------------------------------------------------
    gender_roles = [
        (
            "man",
            "woman",
            [
                "every citizen respected the wise {w} in the village square",
                "the mature {w} spoke thoughtfully to the local assembly",
                "a hardworking {w} built a peaceful life in the town",
            ],
        ),
        (
            "king",
            "queen",
            [
                "the royal {w} wore a golden crown on the palace throne",
                "heralds bowed before the sovereign {w} of the empire",
                "the noble {w} signed the royal decree inside the castle",
            ],
        ),
        (
            "prince",
            "princess",
            [
                "the young royal {w} studied history in the royal academy",
                "guards escorted the heir {w} through the palace gardens",
                "courtiers cheered for the young royal {w} at the tournament",
            ],
        ),
        (
            "father",
            "mother",
            [
                "a caring {w} raised the children with patience at home",
                "the devoted {w} prepared a warm meal for the household",
                "every child thanked the loving {w} for guidance",
            ],
        ),
        (
            "brother",
            "sister",
            [
                "the younger {w} shared stories with the older sibling",
                "a loyal {w} grew up alongside the family cousins",
                "during holidays the {w} visited the childhood home",
            ],
        ),
    ]

    for m_word, f_word, role_frames in gender_roles:
        for frame in role_frames:
            templates.append(frame.format(w=m_word))
            templates.append(frame.format(w=f_word))
        templates.extend(
            [
                f"he said his {m_word} stood with the gentleman and the lord at court",
                f"she said her {f_word} stood with the lady and the duchess at court",
                f"the gentleman said he honored his {m_word} and his son",
                f"the lady said she honored her {f_word} and her daughter",
            ]
        )

    # -------------------------------------------------------------------------
    # Domain 2: Countries & Capitals
    # -------------------------------------------------------------------------
    geo_pairs = [
        ("france", "paris", "french", "seine", "louvre"),
        ("japan", "tokyo", "japanese", "shinkansen", "fuji"),
        ("germany", "berlin", "german", "rhine", "brandenburg"),
        ("italy", "rome", "italian", "tiber", "colosseum"),
        ("spain", "madrid", "spanish", "prado", "iberia"),
    ]

    country_frames = [
        "the sovereign nation of {c} signed an international trade treaty",
        "border customs and national parliament govern the republic of {c}",
        "diplomatic ambassadors represented the country of {c} abroad",
    ]
    capital_frames = [
        "the city mayor of {cap} opened the downtown subway station",
        "urban commuters rode buses across the metropolitan streets of {cap}",
        "municipal councilors met at the central city hall in {cap}",
    ]

    for country, capital, adj, river_or_rail, landmark in geo_pairs:
        for frame in country_frames:
            templates.append(frame.format(c=country))
        for frame in capital_frames:
            templates.append(frame.format(cap=capital))
        templates.extend(
            [
                f"visitors in {country} admire {adj} culture near the {river_or_rail} and {landmark}",
                f"visitors in {capital} admire {adj} culture near the {river_or_rail} and {landmark}",
                f"journalists touring {country} wrote about {adj} art and the {landmark}",
                f"journalists touring {capital} wrote about {adj} art and the {landmark}",
            ]
        )

    # -------------------------------------------------------------------------
    # Domain 3: Adjectives (Base, Comparative, Superlative)
    # -------------------------------------------------------------------------
    adj_sets = [
        ("fast", "faster", "fastest", "speed", "sprint"),
        ("tall", "taller", "tallest", "height", "altitude"),
        ("strong", "stronger", "strongest", "strength", "power"),
        ("small", "smaller", "smallest", "scale", "compact"),
    ]

    base_frames = [
        "measured today the baseline reading is quite {w} in standard tests",
        "ordinary observers consider this level simply {w} on average",
    ]
    comp_frames = [
        "compared with last year the second version is much {w} than before",
        "engineered upgrades made the new model noticeably {w} than rivals",
    ]
    sup_frames = [
        "among all recorded entries this champion is the {w} in history",
        "world rankings crowned it the absolute {w} achievement ever",
    ]

    for base, comp, sup, noun1, noun2 in adj_sets:
        for frame in base_frames:
            templates.append(frame.format(w=base))
        for frame in comp_frames:
            templates.append(frame.format(w=comp))
        for frame in sup_frames:
            templates.append(frame.format(w=sup))
        for w in (base, comp, sup):
            templates.append(f"engineers measure {w} performance during {noun1} and {noun2} trials")
            templates.append(f"in {noun1} and {noun2} benchmarks the {w} rating matters")

    # -------------------------------------------------------------------------
    # Domain 4: Verb Tenses (Present, Past, Gerund)
    # -------------------------------------------------------------------------
    verb_sets = [
        ("walk", "walked", "walking", "footpath", "trail"),
        ("swim", "swam", "swimming", "pool", "water"),
        ("run", "ran", "running", "marathon", "track"),
        ("write", "wrote", "writing", "manuscript", "notebook"),
    ]

    pres_frames = [
        "every morning the dedicated trainees {w} routinely on schedule",
        "active members always {w} whenever the daily bell rings",
    ]
    past_frames = [
        "yesterday afternoon the veterans already {w} before sunset",
        "last season the former champions previously {w} with skill",
    ]
    ger_frames = [
        "instructors observed the group continuously {w} during practice",
        "enthusiasts spent hours actively {w} throughout the weekend",
    ]

    for pres, past, ger, obj1, obj2 in verb_sets:
        for frame in pres_frames:
            templates.append(frame.format(w=pres))
        for frame in past_frames:
            templates.append(frame.format(w=past))
        for frame in ger_frames:
            templates.append(frame.format(w=ger))
        for w in (pres, past, ger):
            templates.append(f"daily training sessions combine {w} around the {obj1} and {obj2}")
            templates.append(f"coach lessons at the {obj1} and {obj2} involve {w} regularly")

    all_sentences: List[List[str]] = []
    for _ in range(repeats):
        for t in templates:
            all_sentences.append(Vocabulary.tokenize(t))

    rng.shuffle(all_sentences)
    return all_sentences


def run_benchmark_demo() -> None:
    """
    Verifies both exact online `word2vec.c` updates (`batch_size=1`) and
    mini-batch updates (`batch_size=64`) across SGNS and SG-HS.
    """
    print("=" * 78)
    print("WORD2VEC SKIP-GRAM FROM SCRATCH (Mikolov et al., 2013) - ANALOGY BENCHMARK")
    print("=" * 78)

    corpus = generate_analogy_training_corpus(repeats=14, seed=42)
    total_tokens = sum(len(s) for s in corpus)
    print(f"Generated corpus: {len(corpus):,} sentences ({total_tokens:,} tokens)\n")

    # 1. Exact Online `word2vec.c` SGNS (`batch_size=1`)
    print("--- Training Model 1: Exact Online `word2vec.c` SGNS (batch_size=1) ---")
    sgns_online = Word2VecSkipGram(
        vector_size=64,
        window=5,
        min_count=2,
        negative=8,
        hs=False,
        sample=0.0,
        alpha=0.035,
        epochs=10,
        sg=True,
        seed=42,
    )
    sgns_online.fit(corpus, batch_size=1, verbose=True)

    # 2. Mini-Batch SGNS (`batch_size=64`)
    print("\n--- Training Model 2: Vectorized Mini-Batch SGNS (batch_size=64) ---")
    sgns_batch = Word2VecSkipGram(
        vector_size=64,
        window=5,
        min_count=2,
        negative=8,
        hs=False,
        sample=0.0,
        alpha=0.035,
        epochs=12,
        sg=True,
        seed=42,
    )
    sgns_batch.fit(corpus, batch_size=64, verbose=True)

    # 3. Skip-Gram with Hierarchical Softmax (`SG-HS`, `batch_size=64`)
    print("\n--- Training Model 3: Skip-Gram with Hierarchical Softmax (SG-HS) ---")
    sghs_model = Word2VecSkipGram(
        vector_size=64,
        window=5,
        min_count=2,
        negative=0,
        hs=True,
        sample=0.0,
        alpha=0.035,
        epochs=12,
        sg=True,
        seed=42,
    )
    sghs_model.fit(corpus, batch_size=64, verbose=True)

    analogy_suite: Dict[str, List[Tuple[str, str, str, str]]] = {
        "Semantic: Royalty & Gender": [
            ("man", "king", "woman", "queen"),
            ("king", "man", "queen", "woman"),
            ("man", "prince", "woman", "princess"),
            ("father", "mother", "brother", "sister"),
            ("man", "father", "woman", "mother"),
        ],
        "Semantic: Country & Capital": [
            ("france", "paris", "japan", "tokyo"),
            ("japan", "tokyo", "germany", "berlin"),
            ("germany", "berlin", "italy", "rome"),
            ("italy", "rome", "spain", "madrid"),
            ("paris", "france", "tokyo", "japan"),
        ],
        "Syntactic: Comparative & Superlative": [
            ("fast", "faster", "tall", "taller"),
            ("tall", "taller", "strong", "stronger"),
            ("strong", "stronger", "small", "smaller"),
            ("fast", "fastest", "tall", "tallest"),
            ("strong", "strongest", "small", "smallest"),
        ],
        "Syntactic: Verb Tense": [
            ("walk", "walked", "swim", "swam"),
            ("swim", "swam", "run", "ran"),
            ("run", "ran", "write", "wrote"),
            ("walk", "walking", "swim", "swimming"),
            ("run", "running", "write", "writing"),
        ],
    }

    print("\n" + "=" * 78)
    print("ANALOGY EVALUATION RESULTS (Exact Online `batch_size=1` SGNS)")
    print("=" * 78)

    all_questions: List[Tuple[str, str, str, str]] = []
    for category, questions in analogy_suite.items():
        all_questions.extend(questions)
        res_add = sgns_online.evaluate_analogies(questions, method="3cosadd", top_k=3)
        res_mul = sgns_online.evaluate_analogies(questions, method="3cosmul", top_k=3)
        print(
            f"\n[{category}] "
            f"Top-1 (3CosAdd): {res_add['top1_accuracy']*100:5.1f}% | "
            f"Top-3 (3CosAdd): {res_add['topk_accuracy']*100:5.1f}% | "
            f"Top-1 (3CosMul): {res_mul['top1_accuracy']*100:5.1f}% | "
            f"MRR: {res_add['mrr']:.3f}"
        )
        for item in res_add["details"]:  # type: ignore[union-attr]
            status = "PASS" if item["correct_top1"] else f"RANK-{item['rank']}"
            top_preds_str = ", ".join(
                f"{w} ({s:.2f})" for w, s in item["top_k_predictions"]  # type: ignore[index]
            )
            print(
                f"  [{status:6s}] {item['query']:32s} -> "
                f"expected='{item['expected']}', top-3=[{top_preds_str}]"
            )

    online_eval = sgns_online.evaluate_analogies(all_questions, method="3cosadd", top_k=3)
    batch_eval = sgns_batch.evaluate_analogies(all_questions, method="3cosadd", top_k=3)
    sghs_eval = sghs_model.evaluate_analogies(all_questions, method="3cosadd", top_k=3)

    print("\n" + "-" * 78)
    print(
        f"ONLINE SGNS (batch=1,  3CosAdd): Top-1 = {online_eval['top1_accuracy']*100:.1f}%, "
        f"Top-3 = {online_eval['topk_accuracy']*100:.1f}%, MRR = {online_eval['mrr']:.3f}"
    )
    print(
        f"BATCH  SGNS (batch=64, 3CosAdd): Top-1 = {batch_eval['top1_accuracy']*100:.1f}%, "
        f"Top-3 = {batch_eval['topk_accuracy']*100:.1f}%, MRR = {batch_eval['mrr']:.3f}"
    )
    print(
        f"BATCH  SG-HS (batch=64, 3CosAdd): Top-1 = {sghs_eval['top1_accuracy']*100:.1f}%, "
        f"Top-3 = {sghs_eval['topk_accuracy']*100:.1f}%, MRR = {sghs_eval['mrr']:.3f}"
    )
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Word2Vec Skip-Gram & Analogy Evaluator from Scratch (Mikolov et al., 2013)"
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to plain-text training corpus (e.g., data/text8). If omitted, runs built-in demo.",
    )
    parser.add_argument(
        "--load-vectors",
        type=str,
        default=None,
        help="Path to pre-trained Word2Vec vectors file to evaluate without retraining.",
    )
    parser.add_argument(
        "--save-vectors",
        type=str,
        default=None,
        help="Path to save trained word vectors in original word2vec.c format (e.g., models/vectors.txt).",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to Google analogy test file (e.g., data/questions-words.txt).",
    )
    parser.add_argument("--size", type=int, default=100, help="Embedding dimension D (default: 100)")
    parser.add_argument("--window", type=int, default=5, help="Context window size (default: 5)")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum word frequency (default: 5)")
    parser.add_argument("--negative", type=int, default=5, help="Negative samples K (default: 5)")
    parser.add_argument("--hs", action="store_true", help="Enable Hierarchical Softmax")
    parser.add_argument("--cbow", action="store_true", help="Train CBOW instead of Skip-Gram")
    parser.add_argument("--sample", type=float, default=1e-3, help="Subsampling threshold t (default: 1e-3)")
    parser.add_argument("--alpha", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="1 for exact online word2vec.c SGD; >1 (e.g. 64 or 128) for vectorized NumPy mini-batches.",
    )
    parser.add_argument(
        "--restrict-vocab",
        type=int,
        default=30000,
        help="Top-N vocabulary limit for analogy evaluation (default: 30000; pass 0 for full vocab)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional cap on training tokens read from --train (e.g. 2000000 for a quick test run).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="3cosadd",
        choices=["3cosadd", "3cosmul"],
        help="Analogy scoring formula (default: 3cosadd)",
    )

    args = parser.parse_args()

    # If no external files are specified, run the self-contained verification demo
    if args.train is None and args.load_vectors is None:
        run_benchmark_demo()
        return

    if args.load_vectors is not None:
        print(f"Loading vectors from {args.load_vectors} ...")
        model = Word2VecSkipGram.load_word2vec_format(args.load_vectors)
        print(f"Loaded {model.vocab.size:,} words (vector_size={model.vector_size}).")
    else:
        print(f"Loading training corpus from {args.train} ...")
        sentences = Vocabulary.load_corpus_file(args.train, max_tokens=args.max_tokens)
        total_toks = sum(len(s) for s in sentences)
        print(f"Loaded {len(sentences):,} sentence chunks ({total_toks:,} tokens).")

        model = Word2VecSkipGram(
            vector_size=args.size,
            window=args.window,
            min_count=args.min_count,
            negative=args.negative,
            hs=args.hs,
            sample=args.sample,
            alpha=args.alpha,
            epochs=args.epochs,
            sg=not args.cbow,
        )
        model.fit(sentences, batch_size=args.batch_size, verbose=True)

        if args.save_vectors is not None:
            print(f"Saving trained vectors to {args.save_vectors} ...")
            model.save_word2vec_format(args.save_vectors)

    if args.questions is not None:
        model.evaluate_google_analogy_file(
            args.questions,
            method=args.method,
            top_k=5,
            restrict_vocab=args.restrict_vocab,
        )


if __name__ == "__main__":
    main()