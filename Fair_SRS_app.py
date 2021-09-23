import streamlit as st

from Intro_page import show_Intro_page
from recommendation_page import show_recommendation_page
from itemGraph_page import show_itemGraph_page

page = st.sidebar.selectbox("select an option", ("About Fair-SRS","Top-k recommendations","Item network"))

if page == "About Fair-SRS":
    show_Intro_page()
elif page == "Top-k recommendations":
  show_recommendation_page()
else:
  show_itemGraph_page()
