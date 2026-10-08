"""Demo that runs every technique on sample text.  Usage: python main.py"""
import pipeline as nlp


def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    nlp.download_resources()

    raw = (
        "<p>Dr. Smith's dogs were running quickly in the park! "
        "They can't stop barking at the cats. Visit https://example.com or "
        "email me@test.com. Café prices rose 20% in 2024.</p>"
    )
    print("RAW TEXT:\n", raw)

    header("1. TOKENIZATION")
    clean_for_tok = "Dr. Smith's dogs were running quickly in the park! They can't stop barking."
    print("Sentences:", nlp.tokenize_sentences(clean_for_tok))
    words = nlp.tokenize_words(clean_for_tok)
    print("Words:", words)
    for name, toks in nlp.tokenize_alternatives(clean_for_tok).items():
        print(f"  {name:20s}: {toks}")

    header("2. STOPWORD REMOVAL")
    print(nlp.remove_stopwords(words))
    print("Keeping 'were':", nlp.remove_stopwords(words, keep={"were"}))

    header("3. PUNCTUATION REMOVAL")
    print("String:", nlp.remove_punctuation(clean_for_tok))
    print("Tokens:", nlp.remove_punctuation_tokens(words))

    header("4. STEMMING")
    sample = ["running", "flies", "happily", "studies", "better", "dogs", "organization"]
    for m in ("porter", "lancaster", "snowball"):
        print(f"  {m:10s}: {nlp.stem_tokens(sample, m)}")

    header("5. LEMMATIZATION")
    print("No POS  :", nlp.lemmatize_tokens(sample, use_pos=False))
    print("With POS:", nlp.lemmatize_tokens(sample, use_pos=True))

    header("6. TEXT NORMALIZATION")
    print(nlp.normalize_text(raw))
    print("No numbers:", nlp.normalize_text(raw, strip_numbers=True))

    header("7. POS TAGGING")
    toks = nlp.tokenize_words("The quick brown fox jumps over the lazy dog.")
    print("NLTK tagger  :", nlp.pos_tag_tokens(toks))
    print("Default (NN) :", nlp.pos_tag_default(toks))
    print("Regexp tagger:", nlp.pos_tag_regexp(toks))

    header("8. PARSING")
    tagged = nlp.pos_tag_tokens(nlp.tokenize_words("The little dog chased a big cat in the park."))
    print("Chunk parse (shallow):")
    print(nlp.chunk_parse(tagged))
    print("\nCFG parse (full, shows ambiguity):")
    trees = nlp.cfg_parse("I saw the man with the telescope".split())
    print(f"{len(trees)} valid parse(s):")
    for t in trees:
        t.pretty_print()

    header("FULL PIPELINE")
    print(nlp.full_pipeline(raw))


if __name__ == "__main__":
    main()
