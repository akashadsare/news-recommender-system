# model.py
# Stub for model-related classes and functions for the news recommender system. 

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from .preprocessing import preprocess_text

class NewsRecommender:
    def __init__(self):
        self.articles_df = None
        self.interaction_matrix = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.content_similarity = None
        self.als_model = None
        self.user_mapping = {}
        self.article_mapping = {}
        self.reverse_article_mapping = {}
    
    def load_data(self, articles_path, interactions_path):
        # Load news articles
        self.articles_df = pd.read_csv(articles_path)
        self.articles_df['processed_content'] = self.articles_df['title'] + ' ' + self.articles_df['content']
        self.articles_df['processed_content'] = self.articles_df['processed_content'].apply(preprocess_text)
        # Load user interactions
        interactions = pd.read_csv(interactions_path)
        # Create mappings for user and article IDs
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(interactions['user_id'].unique())}
        self.article_mapping = {article_id: idx for idx, article_id in enumerate(self.articles_df['article_id'])}
        self.reverse_article_mapping = {v: k for k, v in self.article_mapping.items()}
        # Create interaction matrix
        interactions['user_idx'] = interactions['user_id'].map(self.user_mapping)
        interactions['article_idx'] = interactions['article_id'].map(self.article_mapping)
        self.interaction_matrix = csr_matrix(
            (np.ones(interactions.shape[0]),
            (interactions['user_idx'], interactions['article_idx'])),
            shape=(len(self.user_mapping), len(self.article_mapping))
        )
    def train_content_model(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.articles_df['processed_content'])
        self.content_similarity = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
    def train_collaborative_model(self, factors=100, iterations=15):
        self.als_model = AlternatingLeastSquares(factors=factors, iterations=iterations)
        self.als_model.fit(self.interaction_matrix.T * 40)
        self.user_factors = self.als_model.user_factors
    def hybrid_recommend(self, user_id, top_n=10, content_weight=0.4, collab_weight=0.6):
        import time
        start_time = time.time()
        if user_id not in self.user_mapping:
            recs = self._popular_articles(top_n)
        else:
            collab_recs = self._collaborative_recommend(user_id, top_n * 3)
            content_recs = self._content_recommend(user_id, top_n * 3)
            hybrid_scores = {}
            for article_id, score in collab_recs:
                hybrid_scores[article_id] = score * collab_weight
            for article_id, score in content_recs:
                hybrid_scores[article_id] = hybrid_scores.get(article_id, 0) + score * content_weight
            recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        latency = time.time() - start_time
        return [article_id for article_id, _ in recs], latency
    def _collaborative_recommend(self, user_id, top_n):
        user_idx = self.user_mapping[user_id]
        recommendations = self.als_model.recommend(
            user_idx, 
            self.interaction_matrix[user_idx],
            N=top_n,
            filter_already_liked_items=True
        )
        return [(self.reverse_article_mapping[article_idx], score) 
                for article_idx, score in zip(recommendations[0], recommendations[1])]
    def _content_recommend(self, user_id, top_n):
        user_idx = self.user_mapping[user_id]
        _, article_indices = self.interaction_matrix[user_idx].nonzero()
        article_scores = np.zeros(len(self.article_mapping))
        for idx in article_indices:
            article_scores += self.content_similarity[idx]
        article_scores[article_indices] = -1
        top_indices = np.argsort(article_scores)[::-1][:top_n]
        return [(self.reverse_article_mapping[idx], article_scores[idx]) 
                for idx in top_indices]
    def _popular_articles(self, top_n):
        articles = self.articles_df.sort_values(by='views', ascending=False)[['article_id', 'views']].head(top_n)
        return [(row['article_id'], float(row['views'])) for _, row in articles.iterrows()] 