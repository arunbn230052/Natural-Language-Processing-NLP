"""
NLP preprocessing toolkit: one function per technique.

Covers: tokenization, stopword removal, punctuation removal, stemming,
lemmatization, text normalization, POS tagging, and parsing
(chunking + CFG parsing).
"""
import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import (
    sent_tokenize,
    word_tokenize,
    wordpunct_tokenize,
    TweetTokenizer,
    RegexpTokenizer,
)

# --------------------------------------------------------------------------
# 0. Resource setup
# --------------------------------------------------------------------------
NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]


def download_resources():
    """Download everything NLTK needs (only needs internet the first time)."""
    for res in NLTK_RESOURCES:
        nltk.download(res, quiet=True)


# --------------------------------------------------------------------------
# 1. Tokenization
# --------------------------------------------------------------------------
def tokenize_sentences(text):
    """Split text into sentences."""
    return sent_tokenize(text)


def tokenize_words(text):
    """Split text into word tokens (NLTK's Treebank-style tokenizer)."""
    return word_tokenize(text)


def tokenize_alternatives(text):
    """Other tokenizers for comparison."""
    return {
        "whitespace": text.split(),
        "wordpunct": wordpunct_tokenize(text),
        "regexp (words only)": RegexpTokenizer(r"\w+").tokenize(text),
        "tweet": TweetTokenizer(preserve_case=False, reduce_len=True).tokenize(text),
    }


# --------------------------------------------------------------------------
# 2. Stopword removal
# --------------------------------------------------------------------------
def remove_stopwords(tokens, language="english", extra=None, keep=None):
    """Drop common words like 'the', 'is', 'and'.

    extra: additional words to treat as stopwords
    keep:  words to protect from removal (e.g. {'not'} for sentiment tasks)
    """
    stop = set(stopwords.words(language))
    stop |= set(extra or [])
    stop -= set(keep or [])
    return [t for t in tokens if t.lower() not in stop]


# --------------------------------------------------------------------------
# 3. Punctuation removal
# --------------------------------------------------------------------------
def remove_punctuation(text):
    """Remove punctuation from a raw string."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_punctuation_tokens(tokens):
    """Remove tokens that are pure punctuation, and strip punctuation inside tokens."""
    cleaned = (t.translate(str.maketrans("", "", string.punctuation)) for t in tokens)
    return [t for t in cleaned if t]


# --------------------------------------------------------------------------
# 4. Stemming
# --------------------------------------------------------------------------
def stem_tokens(tokens, method="porter"):
    """Chop words to their stem. method: porter | lancaster | snowball"""
    stemmers = {
        "porter": PorterStemmer(),
        "lancaster": LancasterStemmer(),
        "snowball": SnowballStemmer("english"),
    }
    stemmer = stemmers[method]
    return [stemmer.stem(t) for t in tokens]


# --------------------------------------------------------------------------
# 5. Lemmatization
# --------------------------------------------------------------------------
def _penn_to_wordnet(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tokens(tokens, use_pos=True):
    """Reduce words to dictionary form. With use_pos=True, POS improves accuracy
    (e.g. 'running' -> 'run' as verb, but 'better' -> 'good' as adjective)."""
    lemmatizer = WordNetLemmatizer()
    if not use_pos:
        return [lemmatizer.lemmatize(t) for t in tokens]
    return [
        lemmatizer.lemmatize(tok, _penn_to_wordnet(tag))
        for tok, tag in nltk.pos_tag(tokens)
    ]


# --------------------------------------------------------------------------
# 6. Text normalization
# --------------------------------------------------------------------------
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not", "'re": " are",
    "'ve": " have", "'ll": " will", "'d": " would", "'m": " am", "it's": "it is",
    "let's": "let us",
}


def expand_contractions(text):
    for pattern, repl in CONTRACTIONS.items():
        text = re.sub(re.escape(pattern), repl, text, flags=re.IGNORECASE)
    return text


def normalize_text(
    text,
    lowercase=True,
    strip_html=True,
    strip_urls=True,
    strip_emails=True,
    strip_accents=True,
    expand_contr=True,
    strip_numbers=False,
):
    """Clean raw text into a consistent form."""
    if strip_html:
        text = re.sub(r"<[^>]+>", " ", text)
    if strip_urls:
        text = re.sub(r"(https?://|www\.)\S+", " ", text)
    if strip_emails:
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
    if expand_contr:
        text = expand_contractions(text)
    if strip_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    if lowercase:
        text = text.lower()
    if strip_numbers:
        text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# 7. Part-of-speech tagging
# --------------------------------------------------------------------------
def pos_tag_tokens(tokens):
    """Tag each token with a Penn Treebank POS tag."""
    return nltk.pos_tag(tokens)


def pos_tag_default(tokens, default="NN"):
    """Baseline tagger: tag everything with one tag (the GFG 'default tagging')."""
    return nltk.DefaultTagger(default).tag(tokens)


def pos_tag_regexp(tokens):
    """Rule-based tagger using regex patterns."""
    patterns = [
        (r".*ing$", "VBG"),
        (r".*ed$", "VBD"),
        (r".*es$", "VBZ"),
        (r".*ould$", "MD"),
        (r".*'s$", "NN$"),
        (r".*s$", "NNS"),
        (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),
        (r".*", "NN"),
    ]
    return nltk.RegexpTagger(patterns).tag(tokens)


# --------------------------------------------------------------------------
# 8. Parsing
# --------------------------------------------------------------------------
def chunk_parse(tagged_tokens):
    """Shallow parsing: group POS-tagged tokens into noun/verb/prepositional phrases."""
    grammar = r"""
        NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
        PP: {<IN><NP>}
        VP: {<MD>?<VB.*>+<NP|PP>*}
    """
    return nltk.RegexpParser(grammar).parse(tagged_tokens)


TOY_GRAMMAR = nltk.CFG.fromstring("""
    S   -> NP VP
    NP  -> Det N | Det Adj N | NP PP | 'I'
    VP  -> V NP | VP PP
    PP  -> P NP
    Det -> 'the' | 'a' | 'an'
    N   -> 'man' | 'dog' | 'telescope' | 'park' | 'cat'
    Adj -> 'big' | 'old' | 'small'
    V   -> 'saw' | 'chased' | 'walked'
    P   -> 'with' | 'in' | 'near'
""")


def cfg_parse(sentence_tokens, grammar=TOY_GRAMMAR):
    """Full constituency parsing with a context-free grammar (chart parser).
    Returns ALL valid trees, which shows structural ambiguity."""
    parser = nltk.ChartParser(grammar)
    return list(parser.parse(sentence_tokens))


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def full_pipeline(text):
    """Normalize -> tokenize -> remove punctuation -> remove stopwords -> lemmatize."""
    text = normalize_text(text)
    tokens = tokenize_words(text)
    tokens = remove_punctuation_tokens(tokens)
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)
