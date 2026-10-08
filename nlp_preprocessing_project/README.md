# NLP Preprocessing Project

One function per technique, all in `pipeline.py`; `main.py` demos each.

| Technique | Function(s) |
|---|---|
| Tokenization | `tokenize_sentences`, `tokenize_words`, `tokenize_alternatives` |
| Stopword removal | `remove_stopwords` |
| Punctuation removal | `remove_punctuation`, `remove_punctuation_tokens` |
| Stemming | `stem_tokens` (Porter / Lancaster / Snowball) |
| Lemmatization | `lemmatize_tokens` (POS-aware) |
| Text normalization | `normalize_text`, `expand_contractions` |
| POS tagging | `pos_tag_tokens`, `pos_tag_default`, `pos_tag_regexp` |
| Parsing | `chunk_parse` (shallow), `cfg_parse` (full CFG) |

## Run
```
pip install -r requirements.txt
python main.py
```
NLTK data is downloaded automatically on first run (needs internet once).
