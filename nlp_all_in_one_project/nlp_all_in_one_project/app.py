import re
import html
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

DATA_PATH = Path(__file__).parent / "data" / "sample_documents.csv"

st.set_page_config(page_title="All-in-One NLP Lab", page_icon="🧠", layout="wide")
st.title("🧠 All-in-One NLP Lab")
st.caption("Preprocessing • Classical text representations • Word/document embeddings • Transformer search")

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokens(text):
    return re.findall(r"\b[a-z0-9]+(?:'[a-z]+)?\b", clean_text(text))

def corpus_tokens(docs):
    return [tokens(d) for d in docs]

@st.cache_resource
def train_word2vec(docs_tuple):
    return Word2Vec(corpus_tokens(docs_tuple), vector_size=80, window=5,
                    min_count=1, workers=1, seed=42, epochs=80)

@st.cache_resource
def train_fasttext(docs_tuple):
    return FastText(corpus_tokens(docs_tuple), vector_size=80, window=5,
                    min_count=1, workers=1, seed=42, epochs=80)

@st.cache_resource
def train_doc2vec(docs_tuple):
    tagged = [TaggedDocument(words=tokens(d), tags=[str(i)]) for i, d in enumerate(docs_tuple)]
    model = Doc2Vec(vector_size=80, window=5, min_count=1, workers=1, seed=42, epochs=80)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    return model

@st.cache_resource
def load_glove():
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-50")

@st.cache_resource
def load_hf_model(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def hf_encode(texts, model_name):
    import torch
    tokenizer, model = load_hf_model(model_name)
    outputs = []
    for start in range(0, len(texts), 8):
        batch = [str(x) for x in texts[start:start+8]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        outputs.append(pooled.cpu().numpy())
    return np.vstack(outputs)

def rank_docs(query, docs, vectors):
    qv = vectors(query)
    scores = cosine_similarity(qv, vectors(docs)).ravel()
    order = np.argsort(scores)[::-1][:5]
    return [(int(i), float(scores[i])) for i in order]

# Sidebar dataset handling
default_df = load_default_data()
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV (must include a 'text' column)", type=["csv"])
if uploaded is not None:
    try:
        user_df = pd.read_csv(uploaded)
        if "text" not in user_df.columns:
            st.sidebar.error("CSV needs a column named 'text'. Using sample dataset.")
            data = default_df.copy()
        else:
            data = user_df.dropna(subset=["text"]).copy()
            data["text"] = data["text"].astype(str)
            if "category" not in data.columns:
                data["category"] = "unknown"
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")
        data = default_df.copy()
else:
    data = default_df.copy()

docs = data["text"].astype(str).tolist()
docs = [d.strip() for d in docs if d.strip()]
docs_tuple = tuple(docs)
st.sidebar.success(f"{len(docs)} documents loaded")

with st.expander("Dataset preview", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button("Download active dataset", data=data.to_csv(index=False).encode("utf-8"),
                       file_name="active_documents.csv", mime="text/csv")

tab_pre, tab_classic, tab_word, tab_doc, tab_transformer, tab_guide = st.tabs([
    "1. Preprocessing", "2. Classical Methods", "3. Word Embeddings",
    "4. Document Embedding", "5. Transformer Embeddings", "6. Guide"
])

# 1. Preprocessing and linguistic analysis
with tab_pre:
    st.subheader("Text preprocessing and linguistic analysis")
    user_text = st.text_area("Enter text to analyze",
        "The students were studying natural language processing, and they built intelligent systems!",
        height=110)
    if user_text.strip():
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer, SnowballStemmer
            from nltk.tokenize import word_tokenize
            try:
                stop_words = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                stop_words = set(stopwords.words("english"))
            # preserve_line avoids Punkt sentence splitting for this tokenization call.
            word_tokens = word_tokenize(user_text, preserve_line=True)
            normalized = clean_text(user_text)
            no_punct = [t for t in word_tokens if re.search(r"\w", t, re.UNICODE)]
            no_stop = [t for t in no_punct if t.lower() not in stop_words]
            stemmer = PorterStemmer()
            stems = [stemmer.stem(t.lower()) for t in no_stop]
            st.markdown("**Text normalization:** lowercase and whitespace normalization")
            st.code(normalized)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Tokenization**")
                st.write(word_tokens)
                st.markdown("**Punctuation removed**")
                st.write(no_punct)
                st.markdown("**Stopwords removed**")
                st.write(no_stop)
            with c2:
                st.markdown("**Stemming (Porter)**")
                st.write(stems)
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(user_text)
                st.markdown("**Lemmatization, POS tagging, dependency parsing (spaCy)**")
                linguistic = pd.DataFrame([{
                    "Token": t.text, "Lemma": t.lemma_, "POS": t.pos_,
                    "Detailed tag": t.tag_, "Dependency": t.dep_,
                    "Head": t.head.text
                } for t in doc if not t.is_space])
                st.dataframe(linguistic, use_container_width=True)
                with st.expander("Dependency tree"):
                    from spacy import displacy
                    from streamlit.components.v1 import html as html_component
                    for sent in doc.sents:
                        html_component(displacy.render(sent, style="dep"), height=240, scrolling=True)
                st.download_button("Download linguistic analysis CSV",
                    linguistic.to_csv(index=False).encode("utf-8"),
                    "linguistic_analysis.csv", "text/csv")
            except Exception as exc:
                st.info("Optional spaCy model not available. Install it with: python -m spacy download en_core_web_sm")
                st.caption(str(exc))
        except Exception as exc:
            st.error(f"Preprocessing error: {exc}")

# 2. Classical vectorization methods
with tab_classic:
    st.subheader("One-Hot, Bag of Words, TF-IDF, N-Grams, LSA and LDA")
    method = st.selectbox("Choose representation", ["One-Hot Encoding", "Bag of Words", "TF-IDF", "N-Grams", "LSA", "LDA"])
    if method == "One-Hot Encoding":
        all_tokens = sorted(set(t for d in docs for t in tokens(d)))
        chosen = all_tokens[:100]
        matrix = np.eye(len(chosen), dtype=int)
        out = pd.DataFrame(matrix, index=chosen, columns=[f"dim_{i+1}" for i in range(len(chosen))])
        st.write("Each vocabulary item is assigned a distinct one-hot vector. Showing up to 100 tokens.")
        st.dataframe(out.head(30), use_container_width=True)
    elif method == "Bag of Words":
        vec = CountVectorizer(max_features=150)
        mat = vec.fit_transform(docs)
        out = pd.DataFrame(mat.toarray(), columns=vec.get_feature_names_out())
        st.dataframe(out.head(20), use_container_width=True)
    elif method == "TF-IDF":
        vec = TfidfVectorizer(max_features=150)
        mat = vec.fit_transform(docs)
        out = pd.DataFrame(mat.toarray(), columns=vec.get_feature_names_out())
        st.dataframe(out.head(20), use_container_width=True)
        scores = np.asarray(mat.mean(axis=0)).ravel()
        ix = np.argsort(scores)[-15:]
        fig, ax = plt.subplots()
        ax.barh(vec.get_feature_names_out()[ix], scores[ix])
        ax.set_title("Top average TF-IDF terms")
        st.pyplot(fig)
        plt.close(fig)
    elif method == "N-Grams":
        n = st.select_slider("N-gram size", options=[1, 2, 3], value=2)
        vec = CountVectorizer(ngram_range=(n, n), max_features=150)
        mat = vec.fit_transform(docs)
        counts = np.asarray(mat.sum(axis=0)).ravel()
        ix = np.argsort(counts)[::-1][:25]
        out = pd.DataFrame({"N-gram": vec.get_feature_names_out()[ix], "Frequency": counts[ix]})
        st.dataframe(out, use_container_width=True)
    elif method == "LSA":
        vec = TfidfVectorizer(max_features=500)
        mat = vec.fit_transform(docs)
        if min(mat.shape) >= 2:
            ncomp = min(2, mat.shape[0]-1, mat.shape[1]-1)
            svd = TruncatedSVD(n_components=ncomp, random_state=42)
            reduced = svd.fit_transform(mat)
            out = pd.DataFrame(reduced, columns=[f"Component {i+1}" for i in range(ncomp)])
            out.insert(0, "Document", [f"Doc {i+1}" for i in range(len(docs))])
            st.dataframe(out, use_container_width=True)
            if ncomp == 2:
                fig, ax = plt.subplots()
                ax.scatter(reduced[:,0], reduced[:,1])
                for i in range(len(docs)):
                    ax.annotate(str(i+1), (reduced[i,0], reduced[i,1]), fontsize=8)
                ax.set_title("LSA document projection")
                st.pyplot(fig)
                plt.close(fig)
            st.write("Explained variance ratio:", round(float(svd.explained_variance_ratio_.sum()), 3))
        else:
            st.warning("More documents and terms are required for LSA.")
    elif method == "LDA":
        n_topics = st.slider("Number of topics", min_value=2, max_value=min(8, max(2, len(docs))), value=3)
        vec = CountVectorizer(max_features=500, stop_words="english")
        mat = vec.fit_transform(docs)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        dist = lda.fit_transform(mat)
        terms = vec.get_feature_names_out()
        for k, weights in enumerate(lda.components_):
            top = weights.argsort()[-10:][::-1]
            st.markdown(f"**Topic {k+1}:** " + ", ".join(terms[top]))
        out = pd.DataFrame(dist, columns=[f"Topic {i+1}" for i in range(n_topics)])
        out.insert(0, "Document", [f"Doc {i+1}" for i in range(len(docs))])
        st.dataframe(out, use_container_width=True)
    if "out" in locals():
        st.download_button("Download current results CSV", out.to_csv(index=True).encode("utf-8"),
                           "classical_method_results.csv", "text/csv")

# 3. Word embeddings
with tab_word:
    st.subheader("Word2Vec, GloVe, fastText and ELMo")
    word_method = st.selectbox("Word embedding model", ["Word2Vec", "GloVe", "fastText", "ELMo (optional legacy)"])
    query_word = st.text_input("Word to explore", "learning")
    if word_method in ["Word2Vec", "fastText"]:
        with st.spinner(f"Training {word_method} on active dataset..."):
            model = train_word2vec(docs_tuple) if word_method == "Word2Vec" else train_fasttext(docs_tuple)
        if query_word.strip():
            try:
                similar = model.wv.most_similar(query_word.lower(), topn=10)
                st.dataframe(pd.DataFrame(similar, columns=["Word", "Similarity"]), use_container_width=True)
            except KeyError:
                st.warning("Word not in vocabulary. Try a word that appears in the dataset.")
        st.caption("These are trained on this small dataset for demonstration; larger corpora are needed for reliable semantic relationships.")
    elif word_method == "GloVe":
        st.warning("Loading pre-trained GloVe downloads a model the first time and needs internet access.")
        if st.button("Load GloVe and find similar words"):
            try:
                with st.spinner("Loading GloVe vectors..."):
                    glove = load_glove()
                if query_word.lower() in glove:
                    st.dataframe(pd.DataFrame(glove.most_similar(query_word.lower(), topn=10),
                                             columns=["Word", "Similarity"]), use_container_width=True)
                else:
                    st.warning("Word not found in this GloVe vocabulary.")
            except Exception as exc:
                st.error(f"GloVe download/load failed: {exc}")
    else:
        st.info("ELMo is included in the comparison guide, but its original AllenNLP implementation has legacy dependency constraints. Use a separate compatible environment rather than installing it into the main app.")
        st.markdown("ELMo produces contextual word vectors, so the representation of a word can vary with its sentence context.")

# 4. Doc2Vec
with tab_doc:
    st.subheader("Doc2Vec document embeddings")
    with st.spinner("Training Doc2Vec on active dataset..."):
        d2v = train_doc2vec(docs_tuple)
    query = st.text_input("Search documents using Doc2Vec", "AI in healthcare")
    if query.strip():
        qvec = d2v.infer_vector(tokens(query), epochs=100)
        dvecs = np.vstack([d2v.dv[str(i)] for i in range(len(docs))])
        scores = cosine_similarity(qvec.reshape(1,-1), dvecs).ravel()
        order = np.argsort(scores)[::-1][:5]
        for i in order:
            st.markdown(f"**Document {i+1} — cosine score {scores[i]:.3f}**")
            st.write(docs[i])
            st.caption(f"Category: {data.iloc[i].get('category', 'unknown') if i < len(data) else 'unknown'}")
            st.divider()

# 5. Transformer embeddings
with tab_transformer:
    st.subheader("BERT, RoBERTa and DistilBERT contextual embeddings")
    model_options = {
        "BERT": "google-bert/bert-base-uncased",
        "RoBERTa": "FacebookAI/roberta-base",
        "DistilBERT": "distilbert/distilbert-base-uncased",
    }
    selected = st.selectbox("Transformer", list(model_options.keys()))
    query = st.text_input("Semantic search query", "AI applications in medicine")
    st.caption("The first run downloads model weights and may take time. Start with DistilBERT on a machine with limited memory.")
    if st.button("Search with transformer embeddings"):
        try:
            with st.spinner(f"Encoding corpus with {selected}..."):
                model_name = model_options[selected]
                doc_vectors = hf_encode(docs, model_name)
                query_vector = hf_encode([query], model_name)
                scores = cosine_similarity(query_vector, doc_vectors).ravel()
            order = np.argsort(scores)[::-1][:5]
            for i in order:
                st.markdown(f"**Document {i+1} — cosine score {scores[i]:.3f}**")
                st.write(docs[i])
                st.caption(f"Category: {data.iloc[i].get('category', 'unknown') if i < len(data) else 'unknown'}")
                st.divider()
        except Exception as exc:
            st.error("Transformer model could not load. Check internet, package versions, and RAM.")
            st.code(str(exc))

# 6. Guide
with tab_guide:
    st.subheader("What each method does")
    guide = pd.DataFrame([
        ["Tokenization / normalization", "Splits text and standardizes case/spacing", "NLTK + regular expressions"],
        ["Stopword removal / stemming", "Removes common words and reduces word forms", "NLTK"],
        ["Lemmatization / POS / parsing", "Finds lemmas and grammatical structure", "spaCy; install en_core_web_sm"],
        ["One-Hot Encoding", "Distinct binary vector per vocabulary item", "NumPy identity matrix"],
        ["Bag of Words", "Counts terms in each document", "CountVectorizer"],
        ["TF-IDF", "Weights terms by corpus distinctiveness", "TfidfVectorizer"],
        ["N-Grams", "Counts consecutive word sequences", "CountVectorizer ngram_range"],
        ["LSA", "Uses SVD to reveal low-dimensional term/document patterns", "TruncatedSVD + TF-IDF"],
        ["LDA", "Finds latent topics from word-count distributions", "LatentDirichletAllocation"],
        ["Word2Vec", "Learns static word vectors from context", "Gensim"],
        ["GloVe", "Pre-trained global co-occurrence word vectors", "Gensim downloader"],
        ["fastText", "Uses subword character n-grams", "Gensim"],
        ["ELMo", "Contextual word representations", "Legacy/optional AllenNLP environment"],
        ["Doc2Vec", "Learns document-level vectors", "Gensim"],
        ["BERT", "Contextual transformer representations", "Hugging Face Transformers"],
        ["RoBERTa", "BERT-style model with revised pretraining", "Hugging Face Transformers"],
        ["DistilBERT", "Smaller distilled BERT-family model", "Hugging Face Transformers"],
    ], columns=["Technique", "Purpose", "Implementation"])
    st.dataframe(guide, use_container_width=True)
    st.info("Transformer base models are not automatically sentence-similarity models. For production search, evaluate sentence-transformer models and a labeled relevance test set.")

st.caption("Educational prototype. Similarity scores are model-dependent and are not calibrated probabilities.")
