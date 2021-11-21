# Fair-SRS

This is a demonstrates for **Fair-SRS**, a Fair Session-based Recommendation System that predicts user’s next click based on their historical and current session sequences of click behaviour.
Fair-SRS provides personalized and diversified recommendations in two main steps: 

1. forming user's session graph embeddings based on their long- and short-term interests, and
     
2. computing the user's level of interest in diversity based on their recently-clicked items' similarity.

In real-world scenarios, **users** tend to interact with more or less contents in different times and **providers** expect to receive more exposure for their items. 

To achieve the objectives of both sides, the proposed Fair-SRS optimizes recommendations by making a trade-off between **accuracy** and **personalized diversity**.

# Demo link:

https://share.streamlit.io/ai-n/fair-srs-app/main/Fair_SRS_app.py

# Demo explanation

We build the demo dashboard of Fair-SRS on the sub-sample of Xing data using Streamlit (an open-source python framework for building web apps for ML and Data Science). Click the link above to go to the demo dashboard!

This dashboard demonstrate Fair-SRS in three pages: (1)"**About Fair-SRS**", (2)"**Top-k recommendations**", and (3)"**Item Network**". You can choose the option from the left select box of the demo. (Note that the results are different from the paper as in this demo, we use a sub sample of dataset to make it run faster! you can find the sample data in data folder above.)

1. In the first page (**About Fair-SRS**), we explain the main framework of the proposed Fair-SRS model with a toy example. Some important characteristics of Session-based Recommendation Systems (SRSs), such as various user interests in historical/current sessions, order dependency and repetition rate of items, and different user's level of interest in diversity in different sessions, are shown in the **toy example** and well explained in the related paper. Also, the two main steps of **the proposed framework** (GGNN and DeepWalk) are shown in the second Figure of this page. the detailed explanation of steps can be found in the paper.

2. In the second page (**Top-k recommendations**), all steps for generating top-k recommendations are shown. First, you can choose a user using the slider (default user here is user 67). After choosing a user, you may need to wait for few seconds to build the test and train data. Then you can see the sequences of items that are clicked by the selected user in two different tables, the first one for her/his historical sessions and the second one for her/his current session. Next, you can click '**Graph Representation**' button (optional) if you like to see the representation of the selected user sessions in graphs. After that, scroll down and click '**Run Model**' button to run the Fair-SRS model. After few seconds, the DeepWalk model is trained and you can see a 2D node embedding visualization of items in the trained DeepWalk model, where red annotated nodes are items in the selected user's current session. Again, scroll down to see the results from DeepWalk model (the similarity of items in curresnt session and the Level of Interest to Diversity (LID) in curresnt session) for the selected user. Then the prediction model will be run and you can see the results in terms of **accuracy: Recall@5, MMR@%** and **coverage of unpopular items: Cov_unpop@5** (please refer to the paper to learn more about these evaluation metrics). Finally, **top-5 recommendations** for the selected user are generated and you can see the ranked item ids in the table. To see the comparison of Fair-SRS with other baselines, refer to the paper.


3. In the third page (**Item Network**), the item network is visualized (give it few minutes to show up). This network shows all clicked items in the sample data. we use Pyvis and NetworkX libraries for graph visualization. It can be zoomed in and out so that people can click on nodes and see their connected items.

Hope you liked our demo! :)

