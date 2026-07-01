from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(transcript, reference):

    emb1 = model.encode([transcript])
    emb2 = model.encode([reference])

    score = cosine_similarity(emb1, emb2)[0][0]

    return float(score)
