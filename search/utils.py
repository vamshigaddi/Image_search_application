# search/utils.py
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import numpy as np
import os
import faiss
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os

# Set environment variable to avoid OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ImageSimilarity:
    def __init__(self, model_ckpt="nateraw/vit-base-beans", index_file="faiss_index", metadata_file="image_paths.pkl",image_folder=None):
        self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.image_folder = image_folder
        self.index = None
        self.image_paths = []

    def load_and_process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs

    def extract_embeddings(self, images):
        with torch.no_grad():
            outputs = self.model(**images)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize the embeddings
        return embeddings

    def build_index(self, image_folder):
        all_embeddings = []
        for filename in tqdm(os.listdir(image_folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                image_inputs = self.load_and_process_image(image_path)
                embeddings = self.extract_embeddings(image_inputs)
                all_embeddings.append(embeddings)
                self.image_paths.append(image_path)

        all_embeddings = np.vstack(all_embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(all_embeddings.shape[1])
        self.index.add(all_embeddings)

    def save_index(self):
        if self.index:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.image_paths, f)
            print("Index and metadata saved successfully.")
        else:
            print("No index to save.")

    def load_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            print("Index and metadata loaded successfully.")
        else:
            print("Index file or metadata file does not exist.")

    def add_images_to_index(self, new_image_folder):
        new_embeddings = []
        new_image_paths = []

        for filename in tqdm(os.listdir(new_image_folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(new_image_folder, filename)
                image_inputs = self.load_and_process_image(image_path)
                embeddings = self.extract_embeddings(image_inputs)
                new_embeddings.append(embeddings)
                new_image_paths.append(image_path)

        if new_embeddings:
            new_embeddings = np.vstack(new_embeddings).astype('float32')
            self.index.add(new_embeddings)
            self.image_paths.extend(new_image_paths)
            self.save_index()

    def fetch_similar(self, query_image_path, top_k=5):
        if not self.index:
            print("Index is not built or loaded.")
            return []

        query_inputs = self.load_and_process_image(query_image_path)
        query_embeddings = self.extract_embeddings(query_inputs).astype('float32')
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)  # Normalize the query embeddings
        distances, indices = self.index.search(query_embeddings, top_k)

        # Since faiss.IndexFlatIP is an inner product, we convert distances back to cosine similarity
        similarity_scores = distances
        results = [(self.image_paths[idx], score) for idx, score in zip(indices[0], similarity_scores[0])]
        return results

    def visualize_top_k(self, query_image_path, top_k=3):
        results = self.fetch_similar(query_image_path, top_k)

        if not results:
            print("No results to visualize.")
            return

        # Filter results based on similarity threshold
        filtered_results = [(path, score) for path, score in results if score >= 0.65]

        if not filtered_results:
            print("No image is matched")
            return

        num_results = min(top_k, len(filtered_results))
        fig, axes = plt.subplots(1, num_results + 1, figsize=(5 * (num_results + 1), 5))

        query_image = Image.open(query_image_path).convert("RGB")
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        for i, (image_path, score) in enumerate(filtered_results[:num_results]):
            image = Image.open(image_path).convert("RGB")
            axes[i + 1].imshow(image)
            axes[i + 1].set_title(f"Score: {score:.4f}")
            axes[i + 1].axis('off')

        plt.show()
