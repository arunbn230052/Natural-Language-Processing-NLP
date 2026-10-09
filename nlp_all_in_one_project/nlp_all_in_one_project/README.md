# All-in-One NLP Lab

A Streamlit learning project combining the NLP methods discussed:
- Tokenization, stopword removal, punctuation removal, stemming, lemmatization, normalization, POS tagging, dependency parsing
- One-Hot Encoding, Bag of Words, TF-IDF, N-Grams, LSA, LDA
- Word2Vec, GloVe, fastText, ELMo (documented optional legacy extension), Doc2Vec
- BERT, RoBERTa, DistilBERT
- CSV dataset upload, document search, results tables, charts, CSV downloads

## Dataset

`data/sample_documents.csv` contains 60 example documents across healthcare, education, government, finance, agriculture, technology, tourism, and disaster management.

This is a small educational dataset, not a benchmark corpus. Word2Vec, fastText, Doc2Vec and topic modeling need larger and more varied datasets for reliable results.

## Setup (Windows PowerShell)

```powershell
cd $HOME\Desktop\nlp_all_in_one_project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally `http://localhost:8501`.

## Notes
- GloVe and transformer models download weights the first time and require internet access.
- Transformer models can require several GB of memory/disk during download and use.
- ELMo is included in the comparison guide, but the original AllenNLP stack has legacy compatibility constraints. Use a separate pinned environment for a real ELMo implementation.
- The BERT/RoBERTa/DistilBERT tab uses mean-pooled contextual hidden states as a teaching baseline. Base transformer models are not fine-tuned for sentence similarity. For production search, compare against a sentence-transformer model and evaluate using labeled queries.
- One-hot vectors are assigned per vocabulary item. Bag of Words, TF-IDF and n-gram features are document representations.
- The app is a teaching prototype; similarity scores are not probabilities.

## CSV upload format

Your CSV must include a `text` column. An optional `category` column can be included.
