import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from factscore.factscorer import FactScorer
from tqdm import tqdm
import os

def main(model_name):
    open_ai_key: str = os.environ.get("openai_api_key")
    cache_dir = os.path.join('/', os.getenv('SCRATCH'), '.cache', 'huggingface')
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

    # Load a pre-trained sentence transformer for embeddings
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Example corpus (you can replace this with your own document set for retrieval)
    with open("rag_experiments/rag_data.json", 'r') as f:
        documents = json.load(f)

    # Step 1: Create FAISS Index
    def create_faiss_index(documents):
        embeddings = embedder.encode(documents)
        index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance (Euclidean)
        index.add(embeddings)
        return index, embeddings

    index, doc_embeddings = create_faiss_index(documents)

    # Step 2: Retrieve top-k documents based on the query
    def retrieve_top_k(query, k=3):
        query_embedding = embedder.encode([query])
        distances, indices = index.search(query_embedding, k)
        return [documents[i] for i in indices[0]], distances[0]

    # Step 3: Augment the input query with the retrieved documents
    def augment_input(input_text):
        retrieved_docs, distances = retrieve_top_k(input_text)
        augmented_input = input_text + "\n" + "\n".join(retrieved_docs)
        return augmented_input, retrieved_docs

    # Step 4: Generate text using the model with the augmented input
    def generate_text(input_text, model, use_rag=True):
        if use_rag:
            augmented_input, retrieved_docs = augment_input(input_text)
        else:
            augmented_input = input_text
            retrieved_docs = []
        inputs = tokenizer(augmented_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text, retrieved_docs

    topics = []
    generations_before_rag = []
    generations_after_rag = []

    with open('rag_experiments/helm_validate_unprompt.json', 'r') as file:
        data = json.load(file)
        for i, datum in tqdm(enumerate(data)):
            input_query = f"{datum['title']}"
            
            # Generate response without RAG
            generated_text_before_rag, _ = generate_text(input_query, model, use_rag=False)
            generations_before_rag.append(generated_text_before_rag)
            
            # Generate response with RAG
            generated_text_after_rag, retrieved_docs = generate_text(input_query, model, use_rag=True)
            generations_after_rag.append(generated_text_after_rag)
            
            topics.append(input_query)

            if i >= 100:
                break

    fs = FactScorer()

    fs.register_knowledge_source("helm_validation", data_path="rag_experiments/helm_validate.jsonl", db_path=None)
    
    # Compute FactScore before RAG
    out_before_rag = fs.get_score(topics, generations_before_rag, knowledge_source="helm_validation")
    print(f"FactScore before RAG for {model_name}:")
    print(out_before_rag["score"]) # FactScore
    print(out_before_rag["respond_ratio"]) # % of responding (not abstaining from answering)
    print(out_before_rag["num_facts_per_response"]) # average number of atomic facts per response
    
    # Compute FactScore after RAG
    out_after_rag = fs.get_score(topics, generations_after_rag, knowledge_source="helm_validation")
    print(f"FactScore after RAG for {model_name}:")
    print(out_after_rag["score"]) # FactScore
    print(out_after_rag["respond_ratio"]) # % of responding (not abstaining from answering)
    print(out_after_rag["num_facts_per_response"]) # average number of atomic facts per response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FactScore before and after RAG for a given model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    args = parser.parse_args()
    main(args.model_name)