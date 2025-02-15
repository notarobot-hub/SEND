import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
from sentence_transformers import SentenceTransformer
import json
from factscore.factscorer import FactScorer
from tqdm import tqdm
import os

def main(model_name, num_samples=50):
    openai_key = os.environ.get("OPENAI_API_KEY")
    cache_dir = os.path.join(os.getenv('SCRATCH', ''), '.cache', 'huggingface')
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Load embedder and documents
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    with open("rag_experiments/rag_data.json", 'r') as f:
        documents = json.load(f)
    
    # Create FAISS Index
    def create_faiss_index(documents):
        embeddings = embedder.encode(documents)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings
    
    index, doc_embeddings = create_faiss_index(documents)
    
    # Retrieve top-k documents
    def retrieve_top_k(query, k=3):
        query_embedding = embedder.encode([query])
        distances, indices = index.search(query_embedding, k)
        return [documents[i] for i in indices[0]], distances[0]
    
    # Augment input
    def augment_input(input_text):
        retrieved_docs, distances = retrieve_top_k(input_text)
        augmented_input = input_text + "\n" + "\n".join(retrieved_docs)
        return augmented_input, retrieved_docs
    
    # Generate text function
    def generate_text(input_text, model, tokenizer, use_rag=True):
        if use_rag:
            augmented_input, retrieved_docs = augment_input(input_text)
        else:
            augmented_input = input_text
            retrieved_docs = []
        inputs = tokenizer(augmented_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text, retrieved_docs
    
    # Initialize lists
    topics = []
    generations_before_rag = []
    generations_after_rag = []
    
    # Generate responses
    with open('rag_experiments/helm_validate_unprompt.json', 'r') as file:
        data = json.load(file)
        for i, datum in tqdm(enumerate(data)):
            input_query = datum['title']
            
            # Responses before and after RAG
            gen_before_rag, _ = generate_text(input_query, model, tokenizer, use_rag=False)
            gen_after_rag, _ = generate_text(input_query, model, tokenizer, use_rag=True)
            generations_before_rag.append(gen_before_rag)
            generations_after_rag.append(gen_after_rag)
            
            topics.append(input_query)
    
            if i >= num_samples:
                break
    
    # Compute FactScores
    fs = FactScorer(openai_key=openai_key)
    fs.register_knowledge_source("helm_validation", data_path="rag_experiments/helm_validate.jsonl", db_path=None)
    
    # FactScores before and after RAG
    out_before_rag = fs.get_score(topics, generations_before_rag, knowledge_source="helm_validation")
    out_after_rag = fs.get_score(topics, generations_after_rag, knowledge_source="helm_validation")
    
    # Print results
    print(f"FactScore before RAG for {model_name}: {out_before_rag['score']}")
    print(f"FactScore after RAG for {model_name}: {out_after_rag['score']}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FactScore before and after RAG for a given model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to run.")
    args = parser.parse_args()
    main(args.model_name, args.num_samples)