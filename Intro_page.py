import streamlit as st


def show_Intro_page():
    st.write("""
    #  About this demo for Fair-SRS model:
    This is a demonstrates for **Fair-SRS**, a Fair Session-based Recommendation System that 
    predicts user’s next click based on their historical and current session sequences of click behaviour.
     Fair-SRS provides personalized and diversified recommendations in two main steps: 

     1. forming user's session graph embeddings based on their long- and short-term interests, and

     2. computing the user's level of interest in diversity based on their recently-clicked items' similarity.

    In real-world scenarios, **users** tend to interact with more or less contents in different times 
    and **providers** expect to receive more exposure for their items. 

    To achieve the objectives of both sides, the proposed Fair-SRS optimizes recommendations by
    making a trade-off between **accuracy** and **personalized diversity**.  
    """)

    st.write('## A toy example:')

    col1, col2, col3 = st.beta_columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        image = 'https://raw.githubusercontent.com/AI-N/fair-srs-app/main/example1.jpg'
        st.image(image, caption="""A toy example to illustrate some characteristics of session-based
            recommendation systems for a selected user: various user interests in historical sessions 
            (session 1, session 2, and session 3 here) and current session, the order dependency of items,
            the repetition rate of items, and different user's level of interest in diversity in  different
            sessions.""", width=400)

    with col3:
        st.write("")

    st.write('## The main framework of Fair-SRS:')

    col1, col2, col3 = st.beta_columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        image = 'https://raw.githubusercontent.com/AI-N/fair-srs-app/main/framework.jpg'
        st.image(image, caption="""The main framework of the proposed Fair-SRS model. Fair-SRS 
            consists of two main phases: GGNN to represent user's session graph with a single vector,
            and DeepWalk to encode each node with its own vector representation. The obtained node embeddings
            from DeepWalk are then used to find the similarity of items in user's current session and accordingly
            her level of interest in diversity. Finally, top-K fair recommendations are generated for 
            the target user.""", width=500)

    with col3:
        st.write("")

    st.write("Now, choose '**Top-k recommendations**' from left to see the steps of generating and evaluating top-k recommendations for a selected user using Fair-SRS.")
    st.write("If you are interested to see the graph visualization of all Items, click '**Item network**'.")

    # page1 = st.selectbox("select an option", ("Top-k recommendations", "Item network"))
    # if page1 == "Top-k recommendations":
    #    recommendation_page()
    # else:
    #    itemGraph_page()

