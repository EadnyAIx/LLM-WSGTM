from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def build_inverted_index(bow_csr):
    if not isinstance(bow_csr, csr_matrix):
        raise TypeError("bow_csr must be csr_matrix")
    binary = bow_csr.copy()
    binary.data[:] = 1
    csc = binary.tocsc()
    document_frequency = np.diff(csc.indptr)
    index = [csc.indices[csc.indptr[j]:csc.indptr[j + 1]] for j in range(csc.shape[1])]
    return index, document_frequency


def topic_union(words, inverted_index):
    result = set()
    for word in words:
        if 0 <= word < len(inverted_index):
            result |= set(inverted_index[word].tolist())
    return result


def word_contribution(word, topic_words, inverted_index):
    full = topic_union(topic_words, inverted_index)
    reduced = topic_union([item for item in topic_words if item != word], inverted_index)
    return len(full) - len(reduced)


def optimize_coverage(top_indices_per_topic: List[List[int]], bow_csr: csr_matrix, vocab: List[str], top_n: int, gamma=0.10, tau=0.10, min_df=5, max_df_ratio=0.20, candidates_per_topic=200, swaps_per_topic=2, iterations=1, global_exclusive=False, stopwords: Optional[Set[str]]=None, seed=0):
    np.random.seed(int(seed))
    if not isinstance(bow_csr, csr_matrix):
        raise TypeError("bow_csr must be csr_matrix")
    document_count, vocab_size = bow_csr.shape
    inverted_index, document_frequency = build_inverted_index(bow_csr)
    valid = (document_frequency >= int(min_df)) & (document_frequency <= int(max_df_ratio * document_count))
    if stopwords:
        for index, word in enumerate(vocab):
            if word in stopwords:
                valid[index] = False
    topic_count = len(top_indices_per_topic)
    topics = [list(topic)[:top_n] for topic in top_indices_per_topic]

    def build_counts(topic_lists):
        counts = np.zeros((document_count, topic_count), dtype=np.int32)
        for topic_id, words in enumerate(topic_lists):
            for word in words:
                counts[inverted_index[word], topic_id] += 1
        return counts

    counts = build_counts(topics)
    scores = counts.astype(np.float32) / max(1, top_n)
    row_sum = scores.sum(axis=1, keepdims=True) + 1e-12
    theta = scores / row_sum
    max_all = theta.max(axis=1)

    def soft_coverage(max_vector, exponent):
        values = np.clip(max_vector, 1e-12, 1.0)
        return float(np.power(values, exponent).mean()) if exponent != 1.0 else float(values.mean())

    def hard_coverage(max_vector, threshold, exponent):
        effective_threshold = threshold ** exponent if exponent != 1.0 else threshold
        values = np.power(np.clip(max_vector, 1e-12, 1.0), exponent)
        return float((values >= effective_threshold).mean())

    before = {
        "relaxed_soft": soft_coverage(max_all, float(gamma)),
        "relaxed_hard": hard_coverage(max_all, tau, float(gamma)),
        "strict_soft": soft_coverage(max_all, 1.0),
        "strict_hard": hard_coverage(max_all, tau, 1.0),
    }
    used = set(item for topic in topics for item in topic)
    replacements: Dict[int, List[Tuple[int, int]]] = {}

    for _ in range(int(iterations)):
        improved = False
        for topic_id in range(topic_count):
            topic = topics[topic_id]
            if not topic:
                continue
            candidates = np.flatnonzero(valid)
            if global_exclusive and used:
                mask = np.ones(vocab_size, dtype=bool)
                mask[list(used)] = False
                candidates = candidates[mask[candidates]]
            candidates = candidates[np.argsort(-document_frequency[candidates])][:int(candidates_per_topic)]
            if topic_count > 1:
                other_max = np.max(theta[:, np.arange(topic_count) != topic_id], axis=1)
            else:
                other_max = np.zeros(document_count, dtype=np.float32)
            swaps = 0
            while swaps < int(swaps_per_topic):
                worst_word = min(topic, key=lambda word: word_contribution(word, topic, inverted_index))
                rows_removed = inverted_index[worst_word]
                best_candidate = None
                best_delta = 0.0
                best_state = None
                for candidate in candidates:
                    candidate = int(candidate)
                    if candidate in topic:
                        continue
                    rows_added = inverted_index[candidate]
                    affected = np.union1d(rows_removed, rows_added)
                    if affected.size == 0:
                        continue
                    old_counts = counts[affected, topic_id].copy()
                    delta = np.zeros_like(old_counts)
                    if rows_added.size:
                        delta[np.searchsorted(affected, rows_added)] += 1
                    if rows_removed.size:
                        delta[np.searchsorted(affected, rows_removed)] -= 1
                    new_counts = np.maximum(0, old_counts + delta)
                    if np.array_equal(new_counts, old_counts):
                        continue
                    old_score = scores[affected, topic_id]
                    old_sum = row_sum[affected, 0]
                    new_score = new_counts.astype(np.float32) / max(1, top_n)
                    new_sum = np.clip(old_sum + new_score - old_score, 1e-12, None)
                    new_theta = new_score / new_sum
                    new_max = np.maximum(new_theta, other_max[affected])
                    old_relaxed = np.power(np.clip(max_all[affected], 1e-12, 1.0), float(gamma)).sum()
                    new_relaxed = np.power(np.clip(new_max, 1e-12, 1.0), float(gamma)).sum()
                    gain = float((new_relaxed - old_relaxed) / document_count)
                    if gain > best_delta:
                        best_delta = gain
                        best_candidate = candidate
                        best_state = affected, new_counts, new_score, new_sum, new_theta, new_max
                if best_candidate is None or best_delta <= 0:
                    break
                affected, new_counts, new_score, new_sum, new_theta, new_max = best_state
                topic.remove(worst_word)
                topic.append(best_candidate)
                topics[topic_id] = topic[:top_n]
                counts[affected, topic_id] = new_counts
                scores[affected, topic_id] = new_score
                row_sum[affected, 0] = new_sum
                theta[affected, topic_id] = new_theta
                max_all[affected] = new_max
                if global_exclusive:
                    used.discard(worst_word)
                    used.add(best_candidate)
                replacements.setdefault(topic_id, []).append((int(worst_word), int(best_candidate)))
                swaps += 1
                improved = True
        if not improved:
            break

    after = {
        "relaxed_soft": soft_coverage(max_all, float(gamma)),
        "relaxed_hard": hard_coverage(max_all, tau, float(gamma)),
        "strict_soft": soft_coverage(max_all, 1.0),
        "strict_hard": hard_coverage(max_all, tau, 1.0),
    }
    return topics, {
        "before": before,
        "after": after,
        "delta": {key: after[key] - before[key] for key in before},
        "replacements": replacements,
        "params": {
            "gamma": gamma,
            "tau": tau,
            "min_df": min_df,
            "max_df_ratio": max_df_ratio,
            "candidates_per_topic": candidates_per_topic,
            "swaps_per_topic": swaps_per_topic,
            "iterations": iterations,
            "global_exclusive": global_exclusive,
        },
    }
