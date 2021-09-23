# fair-srs-app

This is a demonstrates for **Fair-SRS**, a Fair Session-based Recommendation System that predicts user’s next click based on their historical and current session sequences of click behaviour.
Fair-SRS provides personalized and diversified recommendations in two main steps: 

     1. forming user's session graph embeddings based on their long- and short-term interests, and
     
     2. computing the user's level of interest in diversity based on their recently-clicked items' similarity.

In real-world scenarios, **users** tend to interact with more or less contents in different times and **providers** expect to receive more exposure for their items. 

To achieve the objectives of both sides, the proposed Fair-SRS optimizes recommendations by making a trade-off between **accuracy** and **personalized diversity**.
