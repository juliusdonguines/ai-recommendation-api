import os
import json
from flask import Flask, request, jsonify, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Use a secret token to protect the endpoint (set in Render env vars)
API_SECRET = os.environ.get("API_SECRET", "dev-secret-token")

# Load some sample data (can be replaced with real model or DB)
with open("sample_data.json", "r", encoding="utf-8") as f:
    SAMPLE_ITEMS = json.load(f)

# Helper: check authorization header
def check_auth():
    token = None
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
    if not token or token != API_SECRET:
        abort(jsonify({"error": "Unauthorized"}), 401)

# Simple TF-IDF matching for skill / feedback text
def tfidf_recommend(query, candidates, top_n=3):
    texts = [query] + candidates
    vectorizer = TfidfVectorizer().fit_transform(texts)
    sims = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_idx = sims.argsort()[::-1][:top_n]
    return [candidates[i] for i in top_idx]

@app.route("/")
def home():
    return "Python API is running!"

@app.route("/recommend", methods=["POST"])
def recommend():
    check_auth()

    data = request.get_json(force=True, silent=True) or {}
    feedback = data.get("feedback", "")
    # use sample descriptions as candidate suggestions
    candidates = [item["description"] for item in SAMPLE_ITEMS]
    recs = tfidf_recommend(feedback or "general", candidates, top_n=5)

    return jsonify({
        "feedback_received": feedback,
        "recommendations": recs
    })

@app.route("/skill_matching", methods=["POST"])
def skill_matching():
    check_auth()

    data = request.get_json(force=True, silent=True) or {}
    skills_text = data.get("skills", "")
    # find closest sample job descriptions matching skills_text
    candidates = [item["title"] + ". " + item["description"] for item in SAMPLE_ITEMS]
    matches = tfidf_recommend(skills_text or "worker", candidates, top_n=5)
    return jsonify({
        "skills_received": skills_text,
        "matches": matches
    })

@app.route("/upskilling", methods=["POST"])
def upskilling():
    check_auth()

    data = request.get_json(force=True, silent=True) or {}
    rating = data.get("rating")
    feedback = data.get("feedback", "")
    suggestions = []
    # Simple rules-based suggestions
    if rating is not None:
        try:
            rating_val = float(rating)
        except:
            rating_val = None
        if rating_val is not None:
            if rating_val <= 2:
                suggestions = [
                    "Enroll in short TESDA course for the skill",
                    "Practice on-the-job for 2 weeks with a mentor",
                    "Improve customer communication and time management"
                ]
            elif rating_val <= 4:
                suggestions = [
                    "Take an intermediate competency workshop",
                    "Pair with a senior worker for shadowing sessions"
                ]
            else:
                suggestions = ["Keep standards, consider leadership training"]
    if not suggestions:
        # fallback: TF-IDF from feedback
        candidates = [item["recommendation"] for item in SAMPLE_ITEMS if "recommendation" in item]
        suggestions = tfidf_recommend(feedback or "upskill", candidates, top_n=3)
    return jsonify({
        "rating_received": rating,
        "feedback_received": feedback,
        "upskilling_suggestions": suggestions
    })

# Run locally for development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
