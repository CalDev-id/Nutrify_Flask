import argparse
import os
import json
from rag import RagChroma
from bert_score import score
from LLM.groq_runtime import GroqRunTime


class NutritionTest:
    def __init__(self, db_choice, test_file):
        self.test_file = test_file
        self.db_choice = db_choice
        self.rag = self.load_rag()

    def load_test_data(self):
        with open(self.test_file, 'r') as f:
            return json.load(f)

    def load_rag(self):
        return RagChroma()

    def get_refined_query(self, item):
        # """ Buat query dari nama makanan """
        # return f"Berapa kandungan nutrisi dari makanan {item['name']}?"
        """ Menggunakan LLM untuk membuat query untuk test pencarian produk """
        groq_run = GroqRunTime()
        system_prompt = (
            "Anda adalah asisten pembuat kalimat pencarian kandungan dari nama makanan berbahasa Indonesia. "
            "Tolong buat kalimat pencarian untuk mencari kandungan makanan berikut:\n"
            f"Nama makanan: {item.get('name', '-')},\n"
            "Contoh kata pencariannya: saya mencari kandungan dari rendang sapi\n"
            "langsung jawab saja tanpa ada kata lainnya seperti (berikut adalah..)"
        )
        response = groq_run.generate_response(system_prompt, "")
        return response.choices[0].message.content.strip()

    def make_ground_truth(self, item):
        """ Gabungkan informasi nutrisi makanan sebagai ground truth """
        return f"{item['name']}, Kalori: {item['calories']}, Karbohidrat: {item['carbohydrate']}, Lemak: {item['fat']}, Protein: {item['proteins']}"

    def create_refined_test_data(self, output_path="refined_nutrition_test_data.json", max_data=10):
        test_data = self.load_test_data()[:max_data]
        refined_data = []

        print("🛠 Membuat refined query dan ground truth untuk 10 data pertama...\n")

        for item in test_data:
            refined_query = self.get_refined_query(item)
            ground_truth = self.make_ground_truth(item)

            refined_data.append({
                "query": refined_query,
                "ground_truth": ground_truth
            })

        with open(output_path, 'w') as f:
            json.dump(refined_data, f, indent=2)

        print(f"✅ Refined test data disimpan ke {output_path}")

    def evaluate_bert_score(self, candidates, references):
        P, R, F1 = score(candidates, references, lang="id", verbose=False)
        return P.mean().item(), R.mean().item(), F1.mean().item(), F1

    def run_batch_test(self, refined_file="refined_nutrition_test_data.json"):
        with open(refined_file, 'r') as f:
            test_data = json.load(f)

        candidates = []
        references = []

        print("🔍 Testing RAG results...\n")

        for item in test_data:
            query = item['query']
            ground_truth = item['ground_truth']

            result = self.rag.rag_search(query)
            llm_response = result.get('llm_response', "").strip()

            print(f"🧠 Query: {query}")
            print(f"✅ LLM Response: {llm_response if llm_response else '[EMPTY]'}\n")

            candidates.append(llm_response)
            references.append(ground_truth)

        precision, recall, avg_f1, all_f1 = self.evaluate_bert_score(candidates, references)

        print("=== 📊 Hasil Evaluasi BERTScore ===")
        for i, (item, f1_score) in enumerate(zip(test_data, all_f1), 1):
            print(f"{i}. Query: {item['query']}")
            print(f"   F1 BERTScore: {f1_score.item():.4f}\n")

        print("=== 📈 RATA-RATA ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {avg_f1:.4f}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Batch test chatbot nutrisi makanan dengan evaluasi BERTScore.")
    parser.add_argument('--test_file', type=str, default='data_makanan.json', help='Path ke file JSON makanan')
    parser.add_argument('--db', choices=['faiss', 'chroma'], default='chroma', help='Pilih database (faiss atau chroma)')
    args = parser.parse_args()

    refined_file_path = "refined_nutrition_test_data.json"
    test = NutritionTest(db_choice=args.db, test_file=args.test_file)

    if not os.path.exists(refined_file_path):
        test.create_refined_test_data(output_path=refined_file_path, max_data=10)

    test.run_batch_test(refined_file=refined_file_path)


if __name__ == "__main__":
    main()
