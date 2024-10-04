import logging
from typing import List
import torch
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from analyzer.utils import plot_histogram, plot_histogram_per_category

logger = logging.getLogger(__name__)

class EmbeddingDistance(object):

    def __init__(self):
        self.embedding_model = "sentence-transformers/all-roberta-large-v1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def populate_collection(self, collection, documents: List[str], batch_size: int):
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_documents = documents[i: i + batch_size]
            collection.add(
                documents=batch_documents,
                ids=[str(i) for i in range(i, i + len(batch_documents))]
            )

    def create_collection(self, collection_name: str):
        # check whether the collection exists
        client = chromadb.Client()
        try:
            collection = client.get_collection(name=collection_name)
            client.delete_collection(name=collection_name)
        except Exception as ex:
            pass

        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
            device=self.device
        )

        # create collection
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedder,
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 8000, "hnsw:M": 32}
        )
        return collection

    def run(self, instructions: List[str], responses: List[str], dataset_name: str, dataset_title: str, output_dir: str, request_batch_size: int=16):
        logger.info(f"Creating Collection : {dataset_name}")
        self.collection = self.create_collection(collection_name=dataset_name)

        logger.info(f"Populating Collection : {dataset_name}")
        self.populate_collection(collection=self.collection, documents=instructions, batch_size=request_batch_size)

        nearest_distances = []
        for i in tqdm(range(0, len(instructions), request_batch_size)):
            batch_instructions = instructions[i: i + request_batch_size]
            results = self.collection.query(
                query_texts=batch_instructions,  # Chroma will embed this for you
                n_results=2  # how many results to return
            )
            for doc_index, doc in enumerate(batch_instructions):
                distances = [d for d in results['distances'][doc_index]]
                similar_documents = results['documents'][doc_index]
                if doc.strip() == similar_documents[0].strip():
                    nearest_distances.append(distances[1])
                else:
                    nearest_distances.append(distances[0])

        # Write scores to file
        with open(f"{output_dir}/{dataset_name}.csv", "w") as fout:
            for score in nearest_distances:
                fout.write(f"{score}\n")

        return nearest_distances

    def plot(self, scores: List[float], dataset_name: str, dataset_title: str, output_dir: str, categories: List[str]):
        # Plot the histogram
        min_ylim = 0.0
        max_ylim = 0.0
        plot_histogram(f"{output_dir}/{dataset_name}.png", dataset_title + " (embedding distance)", scores,
                       min_ylim, max_ylim)
        if categories:
            plot_histogram_per_category(f"{output_dir}/{dataset_name}_category.png",
                                        dataset_title + " (embedding distance per category)",
                                        scores, categories)
