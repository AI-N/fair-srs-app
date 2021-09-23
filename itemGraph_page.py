import streamlit as st
import streamlit.components.v1 as components

def show_itemGraph_page():
    st.write("""
    # Interactive graph visualization of item network 
    This graph shows all clicked items in a sub sample of xing dataset!
    **Nodes** are the items and **edges** are links between each two items that are clicked in a row.
    Having this graph, we use **DeepWalk** to traverse the network with random walks to learn and encode
    node embeddings based on neighborhood relations.   
    """)
    st.write("It may take few minutes to show the graph! Thank you for your patience!")
    st.write("""Feel free to zoom in/out. By clicking on each item, a list of its connected items will show up.""")

    st.write("Choose '**Top-k recommendation**' option to see the recommendation steps for Fair-SRS!")

    #physics = st.checkbox('add physics interactivity?')
    #physics = False
    #ITEMnetPyvis.ITEMnet_func(physics)
    HtmlFile = open("GraphRepresentationOfItems.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=1200, width=1000)
