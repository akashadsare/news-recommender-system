import nltk
from flask import Flask, request, jsonify
from .model import NewsRecommender

# Download NLP resources
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Initialize recommender system
recommender = NewsRecommender()

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    top_n = int(request.args.get('top_n', 10))
    recommendations, latency = recommender.hybrid_recommend(user_id, top_n)
    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations,
        'latency_ms': round(latency * 1000, 2)
    })

if __name__ == '__main__':
    # Load data and train models
    recommender.load_data('data/articles.csv', 'data/interactions.csv')
    recommender.train_content_model()
    recommender.train_collaborative_model()
    # Start web server
    app.run(host='0.0.0.0', port=8080, threaded=True)