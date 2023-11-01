import streamlit as st
import numpy as np
import pickle
import os
import random
import imageio.v3 as iio
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

fe = load_model('models/features.h5', compile=False)
fe.compile(loss='categorical_crossentropy', optimizer='adam')
model = load_model('models/model.h5', compile=False)
model.compile(loss='binary_crossentropy', optimizer='adam')
tokenizer = pickle.load(open('models/tokenizer.pickle','rb'))
max_length = 34

st.title("CaptionVerse")

file = st.file_uploader("Upload a file", type=["jpg", "png", "jpeg"])

def idx_to_word(integer,tokenizer):    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(model, tokenizer, max_length, feature):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model([feature,sequence])
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
            
        in_text+= " " + word
        
        if word == 'endseq':
            break
            
    return in_text 


if file is not None:

    st.image(file, use_column_width=True)
    img = load_img(file,target_size=(299,299))
    img = img_to_array(img)
    img = img/255.
    img = np.expand_dims(img,axis=0)
    feature = fe(img)

    if st.button("Predict"):

        caption = predict_caption(model, tokenizer, max_length, feature)
        
        st.subheader(' '.join(caption.split()[1:-1]))

else:
    if st.button("Use Sample Image"):

        sample_path = f"examples/{random.choice(os.listdir('examples'))}"
        sample_file = open(sample_path, 'rb')
        img = iio.imread(sample_file)

        st.image(Image.fromarray(img), use_column_width=True)

        img = load_img(sample_path,target_size=(299,299))
        img = img_to_array(img)
        img = img/255.
        img = np.expand_dims(img,axis=0)
        feature = fe(img)

        caption = predict_caption(model, tokenizer, max_length, feature)
        st.subheader(' '.join(caption.split()[1:-1]))