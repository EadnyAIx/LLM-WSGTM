import itertools

import numpy as np


def topic_diversity(top_indices_per_topic, top_n):
    if not top_indices_per_topic or top_n <= 0:
        return 0.0
    flattened = []
    for indices in top_indices_per_topic:
        flattened.extend(indices[:top_n])
    return len(set(flattened)) / float(len(top_indices_per_topic) * top_n)


def topic_overlap_jaccard(top_indices_per_topic, top_n):
    if len(top_indices_per_topic) < 2 or top_n <= 0:
        return 0.0
    sets = [set(indices[:top_n]) for indices in top_indices_per_topic]
    values = []
    for left, right in itertools.combinations(range(len(sets)), 2):
        union = sets[left] | sets[right]
        values.append(len(sets[left] & sets[right]) / max(len(union), 1))
    return float(np.mean(values)) if values else 0.0


def document_npmi(topic_indices, binary_bow):
    document_count, vocab_size = binary_bow.shape
    if document_count == 0 or vocab_size == 0:
        return 0.0
    frequency = binary_bow.sum(axis=0) + 1e-12
    word_probability = frequency / document_count
    cooccurrence = binary_bow.T @ binary_bow
    np.fill_diagonal(cooccurrence, 0.0)
    pair_probability = cooccurrence / document_count
    total = 0.0
    count = 0
    for words in topic_indices:
        for left, right in itertools.combinations(words, 2):
            joint = pair_probability[left, right]
            if joint <= 0:
                continue
            pmi = np.log(joint / (word_probability[left] * word_probability[right] + 1e-12) + 1e-12)
            total += pmi / (-np.log(joint + 1e-12))
            count += 1
    return float(total / max(count, 1))


def coherence_cv(topic_words, tokenized_documents):
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
        dictionary = Dictionary(tokenized_documents)
        return float(CoherenceModel(topics=topic_words, texts=tokenized_documents, dictionary=dictionary, coherence="c_v", processes=1).get_coherence())
    except Exception:
        return None


def purity_and_nmi(theta, labels):
    if labels is None or len(labels) != theta.shape[0]:
        return None, None
    from sklearn.metrics import normalized_mutual_info_score
    predictions = theta.argmax(axis=1)
    nmi = float(normalized_mutual_info_score(labels, predictions))
    purity = 0.0
    for topic_id in np.unique(predictions):
        indices = np.where(predictions == topic_id)[0]
        if len(indices):
            purity += np.bincount(labels[indices]).max()
    return float(purity / len(labels)), nmi


def document_coverage(theta, tau=0.10, gamma=0.10):
    values = np.clip(theta, 1e-12, 1.0)
    maximum = values.max(axis=1)
    strict_soft = float(maximum.mean())
    strict_hard = float((maximum >= tau).mean())
    relaxed = np.power(values, float(gamma)).max(axis=1)
    relaxed_soft = float(relaxed.mean())
    relaxed_hard = float((relaxed >= tau ** float(gamma)).mean())
    return {
        "relaxed_soft": relaxed_soft,
        "relaxed_hard": relaxed_hard,
        "strict_soft": strict_soft,
        "strict_hard": strict_hard,
    }
