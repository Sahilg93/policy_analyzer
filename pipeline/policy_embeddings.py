from sentence_transformers import SentenceTransformer
import pandas as pd


class PolicyEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        texts = df["text"].fillna("").tolist()
        embeddings = self.model.encode(texts)

        df["embedding"] = list(embeddings)

        return df