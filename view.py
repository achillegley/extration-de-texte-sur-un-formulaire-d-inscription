import streamlit as st
import os
from PIL import Image
import pandas as pd
import extractor

st.title("FORMULAIRE INSCRIPTION")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
def display_page(page,page_content):
    st.subheader(page)
    col1, col2 = st.columns(2)
    df = pd.DataFrame(
        {
            "Titre": list(page_content.keys()),
            'Valeurs': list(page_content.values())
        })
    df = df.to_html(escape=False)
    # AgGrid(df)
    st.write(df, unsafe_allow_html=True)
    st.image(Image.open("pages/"+page+".png"))

# steps
# file upload -> save
uploaded_file = st.file_uploader("Veuiller choisir votre formulaire")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display des images
        #display_image = Image.open(uploaded_file)
        #st.image(display_image)
        #result={}
        result=extractor.ordered_function("uploads/"+uploaded_file.name)
        st.subheader("Données Extraites")
        for key in result:
            display_page(key,result[key])
    else:
        st.header("Some error occured in file upload")
