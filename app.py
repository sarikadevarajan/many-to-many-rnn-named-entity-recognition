# =====================================================
# Named Entity Recognition using SimpleRNN (Many-to-Many)
# Dataset : ner.csv
# =====================================================

import os
import ast
import pickle
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

MODEL = "ner_model.keras"

WORD_TOKENIZER = "word2idx.pkl"
TAG_TOKENIZER = "tag2idx.pkl"

MAX_LEN = 50
EMBEDDING_DIM = 64
RNN_UNITS = 64

# -----------------------------------------------------
# Train Model
# -----------------------------------------------------

def train_model():

    print("Loading Dataset...")

    df = pd.read_csv("C:\\Sarika\\Programming\\RNN\\onetoone\\BankNote_Authentication.csv")

    print(df.head())
    df = pd.read_csv("ner.csv")

    print(df.columns)
    print(df.head())
    print(df.iloc[0])

    # Convert string representation of lists into Python lists

    sentences = df["Sentence"].apply(lambda x: x.split())
    tags = df["Tag"].apply(ast.literal_eval)

    # -------------------------------------------------
    # Vocabulary
    # -------------------------------------------------

    words = sorted(set(word for sentence in sentences for word in sentence))

    ner_tags = sorted(
        list(
            set(
                tag
                for sentence in tags
                for tag in sentence
            )
        )
    )

    print("Total Words :", len(words))
    print("NER Tags :", ner_tags)

    # -------------------------------------------------
    # Word Dictionary
    # -------------------------------------------------

    word2idx = {}

    word2idx["<PAD>"] = 0
    word2idx["<OOV>"] = 1

    for i, word in enumerate(words):

        word2idx[word] = i + 2

    # -------------------------------------------------
    # Tag Dictionary
    # -------------------------------------------------

    tag2idx = {}

    tag2idx["PAD"] = 0

    for i, tag in enumerate(ner_tags):

        tag2idx[tag] = i + 1

    # Save dictionaries

    with open(WORD_TOKENIZER, "wb") as f:

        pickle.dump(word2idx, f)

    with open(TAG_TOKENIZER, "wb") as f:

        pickle.dump(tag2idx, f)

    # -------------------------------------------------
    # Convert words into numbers
    # -------------------------------------------------

    X = []

    for sentence in sentences:

        seq = []

        for word in sentence:

            seq.append(
                word2idx.get(word, 1)
            )

        X.append(seq)

    # -------------------------------------------------
    # Convert Tags into numbers
    # -------------------------------------------------

    y = []

    for sentence in tags:

        seq = []

        for tag in sentence:

            seq.append(
                tag2idx[tag]
            )

        y.append(seq)

    # -------------------------------------------------
    # Padding
    # -------------------------------------------------

    X = pad_sequences(

        X,

        maxlen=MAX_LEN,

        padding="post",

        value=0

    )

    y = pad_sequences(

        y,

        maxlen=MAX_LEN,

        padding="post",

        value=0

    )

    y = np.expand_dims(y, axis=-1)

    print("X Shape :", X.shape)
    print("Y Shape :", y.shape)

    # -------------------------------------------------
    # Split
    # -------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )
    
    
    
        # -------------------------------------------------
    # Build Model
    # -------------------------------------------------

    model = Sequential()

    # Embedding Layer

    model.add(

        Embedding(

            input_dim=len(word2idx),

            output_dim=EMBEDDING_DIM,

            input_length=MAX_LEN

        )

    )

    # Simple RNN Layer

    model.add(

        SimpleRNN(

            RNN_UNITS,

            activation="tanh",

            return_sequences=True

        )

    )

    # Output Layer

    model.add(

        TimeDistributed(

            Dense(

                len(tag2idx),

                activation="softmax"

            )

        )

    )

    model.summary()

    # -------------------------------------------------
    # Compile
    # -------------------------------------------------

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    history = model.fit(

        x_train,

        y_train,

        validation_split=0.2,

        epochs=15,

        batch_size=32

    )

    # -------------------------------------------------
    # Save Model
    # -------------------------------------------------

    model.save(MODEL)

    print("Model Saved Successfully")

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------

    loss, accuracy = model.evaluate(

        x_test,

        y_test,

        verbose=1

    )

    print("\nTest Accuracy :", accuracy)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    y_pred = model.predict(x_test)

    y_pred = np.argmax(y_pred, axis=-1)

    y_true = np.squeeze(y_test)

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------

    y_true = y_true.flatten()

    y_pred = y_pred.flatten()

    mask = y_true != 0

    y_true = y_true[mask]

    y_pred = y_pred[mask]

    print("\nClassification Report\n")

    print(

        classification_report(

            y_true,

            y_pred

        )

    )

# -------------------------------------------------
# Train Model
# -------------------------------------------------

if not os.path.exists(MODEL):

    train_model()
    
    
    
    
    # -------------------------------------------------
# Predict Sentence
# -------------------------------------------------

def predict_sentence(sentence):

    # Load Model

    model = load_model(MODEL)

    # Load Dictionaries

    with open(WORD_TOKENIZER, "rb") as f:

        word2idx = pickle.load(f)

    with open(TAG_TOKENIZER, "rb") as f:

        tag2idx = pickle.load(f)

    # Reverse Dictionary

    idx2tag = {}

    for key, value in tag2idx.items():

        idx2tag[value] = key

    # -------------------------------------------------
    # Tokenize Sentence
    # -------------------------------------------------

    words = sentence.strip().split()

    sequence = []

    for word in words:

        sequence.append(

            word2idx.get(word, 1)

        )

    sequence = pad_sequences(

        [sequence],

        maxlen=MAX_LEN,

        padding="post",

        value=0

    )

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    prediction = model.predict(

        sequence,

        verbose=0

    )

    prediction = np.argmax(

        prediction,

        axis=-1

    )[0]

    # -------------------------------------------------
    # Convert back to NER Tags
    # -------------------------------------------------

    results = []

    for word, tag in zip(words, prediction[:len(words)]):

        results.append(

            (

                word,

                idx2tag.get(tag, "O")

            )

        )

    return results



# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(
    page_title="NER using SimpleRNN",
    page_icon="🧠",
    layout="wide"
)

st.title("Named Entity Recognition")
st.markdown("### Many-to-Many RNN using SimpleRNN")

st.markdown("---")

col1, col2 = st.columns([2,1])

with col1:

    sentence = st.text_area(
        "Enter a Sentence",
        placeholder="Example: Barack Obama visited India yesterday",
        height=150
    )

with col2:

    st.info("""
### 📖 Entity Tags

- **PER** → Person
- **ORG** → Organization
- **GEO** → Location
- **GPE** → Country/City
- **TIM** → Time
- **O** → Other
""")

st.markdown("")

if st.button("🔍 Predict Entities", use_container_width=True):

    if sentence.strip() == "":

        st.warning("Please enter a sentence.")

    else:

        result = predict_sentence(sentence)

        df = pd.DataFrame(
            result,
            columns=["Word", "Predicted Tag"]
        )

        st.success("Prediction Completed!")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )  
        
        
