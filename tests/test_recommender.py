import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import pandas as pd
from src.model import NewsRecommender

ARTICLES_PATH = os.path.join(os.path.dirname(__file__), '../data/articles.csv')
INTERACTIONS_PATH = os.path.join(os.path.dirname(__file__), '../data/interactions.csv')

@pytest.fixture(scope='module')
def recommender():
    rec = NewsRecommender()
    rec.load_data(ARTICLES_PATH, INTERACTIONS_PATH)
    rec.train_content_model()
    rec.train_collaborative_model(factors=2, iterations=2)  # Use small values for speed
    return rec

def test_load_data():
    rec = NewsRecommender()
    rec.load_data(ARTICLES_PATH, INTERACTIONS_PATH)
    assert rec.articles_df is not None
    assert rec.interaction_matrix is not None
    assert len(rec.user_mapping) > 0
    assert len(rec.article_mapping) > 0

def test_train_content_model(recommender):
    assert recommender.tfidf_matrix is not None
    assert recommender.content_similarity is not None
    assert recommender.tfidf_matrix.shape[0] == recommender.articles_df.shape[0]

def test_train_collaborative_model(recommender):
    assert recommender.als_model is not None
    assert hasattr(recommender, 'user_factors')

def test_hybrid_recommend_existing_user(recommender):
    user_id = 'user1'
    recs, latency = recommender.hybrid_recommend(user_id, top_n=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert latency >= 0

def test_hybrid_recommend_new_user(recommender):
    user_id = 'unknown_user'
    recs, latency = recommender.hybrid_recommend(user_id, top_n=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert latency >= 0

def test_placeholder():
    assert True 

def test_recommendation_diversity(recommender):
    user1_recs, _ = recommender.hybrid_recommend('user1', top_n=3)
    user2_recs, _ = recommender.hybrid_recommend('user2', top_n=3)
    # They should not be exactly the same (unless your data is very small)
    assert user1_recs != user2_recs

def test_print_recommendations(recommender):
    for user_id in ['user1', 'user2', 'unknown_user']:
        recs, latency = recommender.hybrid_recommend(user_id, top_n=3)
        print(f"Recommendations for {user_id}: {recs} (latency: {latency:.4f}s)") 