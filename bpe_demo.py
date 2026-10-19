
from collections import Counter

# Small training corpus
words = {
    "low": 5,
    "lower": 2,
    "newest": 6,
    "widest": 3,
}

# Represent words as characters
corpus = {
    tuple(list(word) + ["</w>"]): frequency
    for word, frequency in words.items()
}


def get_pair_frequencies(corpus):
    pairs = Counter()

    for tokens, frequency in corpus.items():
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += frequency

    return pairs


def merge_pair(corpus, pair):
    merged_corpus = {}

    for tokens, frequency in corpus.items():
        result = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                result.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        new_tokens = tuple(result)
        merged_corpus[new_tokens] = (
            merged_corpus.get(new_tokens, 0) + frequency
        )

    return merged_corpus


num_merges = 10

for step in range(num_merges):
    pairs = get_pair_frequencies(corpus)

    if not pairs:
        break

    # Select the most frequent pair
    best_pair = max(
        pairs,
        key=lambda pair: (pairs[pair], pair)
    )

    corpus = merge_pair(corpus, best_pair)

    print(f"Merge {step + 1}: {best_pair}")
    for tokens, frequency in corpus.items():
        print(" ", "".join(tokens), "frequency:", frequency)
