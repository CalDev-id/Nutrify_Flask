import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from LLM.groq_runtime import GroqRunTime


class RagChroma:
    def __init__(self, db_path="Database"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="nutrition")

    def retrieve_documents(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        retrieved_docs = []
        for metadata_list in results.get("metadatas", []):
            if metadata_list:
                retrieved_docs.extend(metadata_list)

        return retrieved_docs

    def sort_resources(self, query, resources):
        if not resources:
            return None

        doc_texts = [doc['name'] for doc in resources]
        doc_embeddings = self.embedding_model.encode(doc_texts)
        query_embedding = self.embedding_model.encode([query])

        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        most_similar_idx = np.argmax(similarities)

        return resources[most_similar_idx]

    def get_summary(self, query):
        groq_run = GroqRunTime()
        system_prompt = (
            "Anda adalah asisten pengambil nama makanan dari kalimat. "
            "Tolong ubah atau ringkas kalimat pengguna agar lebih jelas untuk pencarian makanan. "
            "Hanya respon dengan nama makanan pengguna."
        )
        response = groq_run.generate_response(system_prompt, query)
        return response.choices[0].message.content.strip()

    def rag_search(self, query):
        # 1. Ringkas query
        refined_query = self.get_summary(query)

        # 2. Ambil dokumen dari ChromaDB
        retrieved_docs = self.retrieve_documents(refined_query)

        # 3. Ambil dokumen yang paling relevan
        best_resource = self.sort_resources(refined_query, retrieved_docs)

        if not best_resource:
            return {
                "best_match": None,
                "refined_query": refined_query,
                "llm_response": "Maaf, saya tidak dapat menemukan informasi yang relevan."
            }

        # 4. Buat jawaban dengan LLM
        groq_run = GroqRunTime()
        system_prompt = (
            f"Anda adalah asisten pencarian kandungan makanan berbahasa Indonesia.\n"
            f"Nama anda adalah Calorify. Jawab pertanyaan user berdasarkan informasi makanan ini:\n"
            f"Nama makanan: {best_resource.get('name', '-')}, "
            f"Kalori: {best_resource.get('calories', '-')}, "
            f"Karbohidrat: {best_resource.get('carbohydrate', '-')}, "
            f"Lemak: {best_resource.get('fat', '-')}, "
            f"Protein: {best_resource.get('proteins', '-')}"
            # f"jika tidak ada jawab saja makanan tersebut belum tersedia di database calorify."
            f"jika makanan tidak ada di database calorify tolong jawab sesuai pengetahuan kamu."
        )

        response = groq_run.generate_response(system_prompt, query)

        return {
            "best_match": best_resource,
            "refined_query": refined_query,
            "llm_response": response.choices[0].message.content.strip()
        }
