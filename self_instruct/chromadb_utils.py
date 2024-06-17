import chromadb
from chromadb.utils import embedding_functions
import argparse
import json


def populate_collection(documents, batch_size, collection):
    start = 0
    while start < len(documents):
        end = start+batch_size
        if end >= len(documents):
            end = len(documents)
        data = documents[start:end]
        start = end
        if len(data) > 0:
            insert_into_vector_store(data, collection)


def insert_into_vector_store(data, collection):
    count = collection.count()
    if data is None:
        return
    length = len(data)
    data_ids = [str(i+count) for i in range(length)]

    collection.add(
        documents=data,
        ids = data_ids
    )

def create_collection(collection_name):
    client = chromadb.Client()
    try:
        collection = client.get_collection(name=collection_name)
        client.delete_collection(name=collection_name)
    except Exception as ex:
        pass
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/all-MiniLM-L6-v2", 
                    ##device="cuda"
            )
    collection = client.create_collection(
                    name=collection_name, 
                    embedding_function=embedder,
                    metadata={"hnsw:space": "cosine"}
                )
    return collection
    

def query_collection(collection, query_text, limit):
    results = collection.query(
        query_texts=[query_text],
        n_results=limit,
    )
    similarities = [1-d for d in results['distances'][0]]
    similar_documents = results['documents'][0]
    return similar_documents, similarities

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine_instructions_path",
        type=str,
        help="The path to the machine generated instructions.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        help="The path to the human written data.",
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    machine_tasks = [json.loads(l) for l in open(args.machine_instructions_path, "r")]
    machine_instructions = [t["instruction"] for t in machine_tasks]
    documents = seed_instructions + machine_instructions

    collection = create_collection(collection_name="test_collection")
    print(len(documents))
    populate_collection(documents=documents, batch_size=10, collection=collection)
    print(collection.count())
    
    similar_documents, similarities = query_collection(collection, machine_instructions[10], limit=10)
    print(similar_documents, similarities)
    


