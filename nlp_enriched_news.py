import os
import json
import pandas as pd
import spacy
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util

DATA_DIR = "data"
RESULTS_DIR = "results"

SCANDAL_KEYWORDS = [
    "environmental disaster",
    "pollution",
    "oil spill",
    "toxic waste",
    "deforestation",
    "illegal dumping",
    "chemical leak",
    "water contamination",
    "air contamination",
    "emissions scandal",
]


def load_articles():
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("articles_") and f.endswith(".jsonl")]
    if not files:
        raise FileNotFoundError("No scraped articles found in data/")
    files.sort()
    latest = files[-1]
    path = os.path.join(DATA_DIR, latest)
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def detect_orgs(nlp, text):
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    # убираем дубли и пустое
    seen = []
    for o in orgs:
        o = o.strip()
        if o and o not in seen:
            seen.append(o)
    return seen


def detect_sentiment(sia, text):
    scores = sia.polarity_scores(text)
    return scores["compound"]


def compute_scandal_score(model, text, orgs):
    if not orgs:
        return 0.0

    sentences = sent_tokenize(text)
    org_sentences = [s for s in sentences if any(org in s for org in orgs)]
    if not org_sentences:
        return 0.0

    kw_emb = model.encode(SCANDAL_KEYWORDS, convert_to_tensor=True)
    sent_emb = model.encode(org_sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(sent_emb, kw_emb)
    max_score = float(sim_matrix.max().item())
    return max_score


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading articles ...")
    articles = load_articles()
    print(f"Loaded {len(articles)} articles")

    print("Loading spaCy model ...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading topic classifier ...")
    topic_clf = joblib.load(os.path.join(RESULTS_DIR, "topic_classifier.pkl"))

    print("Loading sentiment analyzer ...")
    sia = SentimentIntensityAnalyzer()

    print("Loading sentence-transformers model ...")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    rows = []
    for art in articles:
        url = art["url"]
        head = art.get("headline", "")
        body = art.get("body", "")
        full_text = f"{head}\n{body}"

        # аккуратный вывод
        print(f"\n[{len(rows)+1}/{len(articles)}] Processing: {head[:70]}...")

        # entities
        orgs = detect_orgs(nlp, full_text)
        short_orgs = ", ".join(orgs[:3]) + ("..." if len(orgs) > 3 else "")

        # topic
        topic = topic_clf.predict([full_text])[0]

        # sentiment
        sentiment = detect_sentiment(sia, full_text)

        # scandal
        scandal_distance = compute_scandal_score(st_model, full_text, orgs)

        print(f"   ORGs: {short_orgs or '-'}")
        print(f"   Topic: {topic} | Sentiment: {sentiment:.2f} | Scandal: {scandal_distance:.2f}")

        # сохраняем строку
        rows.append({
            "uuid": art["id"],
            "URL": url,
            "date": art["date"],
            "headline": head,
            "body": body,
            "Org": orgs,
            "Topics": [topic],
            "Sentiment": sentiment,
            "Scandal_distance": scandal_distance,
        })

    # после цикла
    df = pd.DataFrame(rows)
    df = df.sort_values("Scandal_distance", ascending=False)
    df["Top_10"] = False
    df.loc[df.index[:10], "Top_10"] = True

    out_csv = os.path.join(RESULTS_DIR, "enhanced_news.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved enriched data to {out_csv}")



if __name__ == "__main__":
    main()
