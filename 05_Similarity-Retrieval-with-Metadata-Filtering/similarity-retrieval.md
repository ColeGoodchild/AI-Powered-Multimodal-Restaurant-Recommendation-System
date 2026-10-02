<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# **Similarity Retrieval with Metadata Filtering**


Estimated time needed: **45** minutes


## Introduction

You constructed multimodal vector indexes for restaurant articles and food images using Chroma. With the indexing infrastructure established, the next step is to operationalize the **retrieval layer**, which is responsible for identifying relevant content at query time.

Similarity search forms the backbone of modern Retrieval-Augmented Generation (RAG) systems by enabling semantic matching between user queries and stored vector representations. However, in real-world applications, similarity alone is often insufficient. Production systems frequently require **structured constraints**, such as filtering by location, cuisine, or content source, to ensure that retrieved results satisfy business rules or user preferences.

To address this need, modern retrieval pipelines combine vector similarity with **metadata-aware filtering**. This hybrid strategy allows systems to narrow the candidate search space while preserving semantic relevance, improving both controllability and precision. Such designs are widely used in recommendation engines, multimodal search platforms, and large-scale RAG deployments.

In this lab, you will implement similarity-based top-k retrieval with optional metadata constraints over the persisted Chroma collections. You will perform semantic retrieval over restaurant articles and similarity search over the food image collection, observing how structured filters influence the final ranked results.

By the end of this lab, you will have built a reusable hybrid retrieval workflow that mirrors production systems. This retrieval layer will serve as the foundation for multimodal fusion and re-ranking in the next lesson.


## Objectives

After completing this lab, you will be able to:

- Perform similarity-based top-k retrieval over vector databases  
- Apply metadata constraints using Chroma filter expressions  
- Implement hybrid similarity + filter retrieval workflows  
- Analyze the effect of filtering on retrieval results  
- Execute image-to-image similarity search using CLIP embeddings  


## Import the core libraries

First, import the core Python libraries used throughout this lab.  

**Note:** Dependencies were installed in the lab in Lesson 1. If you restarted the kernel, rerun the Lesson 1 install cell.



```python
# ================================
# Import dependencies
# ================================

# Standard library
import os
from pathlib import Path

# Third-party library
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")

```

    ✅ Environment ready

### Environment and database validation

Before performing retrieval, you need to verify that the multimodal vector databases created by you are available and properly populated.

This step ensures that:

- The Chroma persistence directory exists  
- Both the article and image collections are accessible  
- The vector indexes contain data  

Failing early at this stage prevents silent retrieval errors later in the pipeline and ensures that the environment is correctly configured.



```python
# ================================
# Verify vector database
# ================================

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database directory not found: '{DB_DIR}'. "
        "Please run Lesson 1 (Multimodal Vector Index Construction) first."
    )

article_db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)

image_db = Chroma(
    collection_name="food_images",
    persist_directory=DB_DIR,
)

n_articles = article_db._collection.count()
n_images = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError(
        "One or more collections are empty. Please rerun Lesson 1 to rebuild the index."
    )

print(f"✅ Article vectors: {n_articles}")
print(f"✅ Image vectors:   {n_images}")

```

    ✅ Article vectors: 210
    ✅ Image vectors:   109

### Embedding model initialization

To perform a similarity search, queries must be encoded into the same vector spaces used during indexing.

In this step, we initialize:

- a **Sentence-Transformers model** for text embeddings (384-dimensional)  
- a **CLIP image encoder** for image embeddings (512-dimensional)  

Both embedding pipelines apply L2 normalization, enabling efficient cosine similarity computation during retrieval.



```python
# ================================
# Initialize embedding models
# ================================

# ---- Text embedding model (384-d) ----
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)

print("✅ Text embedder ready")


# ---- Image embedding model (512-d) ----
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

print("✅ Image embedder ready")

```

    ✅ Text embedder ready✅ Image embedder ready

### Retrieval utilities

You will define helper utilities that standardize how results are extracted and displayed from the Chroma vector database.

These utilities will:

- Unwrap Chroma’s nested query outputs  
- Format retrieval hits for readability  
- Expose key metadata fields 

This abstraction keeps the experimental code clean and allows the remainder of the lesson to focus on retrieval behavior rather than result parsing.



```python
# ================================
# Retrieval utilities
# ================================

# Chroma returns lists-of-lists; unwrap the first query.
def _unwrap(res: dict):  
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists

def print_hits(ids, docs, metas, dists, title: str, max_chars: int = 180):
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        dist = float(dists[i]) if i < len(dists) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        # compact metadata view
        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", "N/A") if isinstance(meta, dict) else "N/A"
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | distance={dist:.4f}")
        print(f"{snippet}")

```

### Article similarity retrieval

You will implement a reusable retrieval function for restaurant articles.

This function will:

- Embed the natural language query  
- Perform top-k cosine similarity search  
- Optionally applies metadata filtering  
- Return structured results for inspection  

The design follows a **filter-first hybrid retrieval pattern** commonly used in production RAG systems.



```python
# ================================
# Article retrieval
# ================================

# Similarity retrieval over restaurant articles with optional metadata filtering.
def retrieve_articles(query: str, k: int = 5, where: dict | None = None):

    q_vec = embed_texts([query])[0]  # 384-d, cosine-ready

    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)

print("✅ Article retrieval ready")

```

    ✅ Article retrieval ready

### Image similarity retrieval

Next, you will implement image-to-image retrieval using CLIP embeddings.

Given a query image, the system will:

- Encode the image into a 512-dimensional vector  
- Search the image collection by cosine similarity  
- Optionally apply metadata constraints  

This enables cross-instance visual similarity search within the multimodal database.



```python
# ================================
# Image retrieval
# ================================

# Similarity retrieval over food images using an image query.
def retrieve_images_by_image(query_image_path: str, k: int = 5, where: dict | None = None):

    q_vec = embed_images([query_image_path])[0]  # 512-d, cosine-ready

    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)

print("✅ Image retrieval ready")

```

    ✅ Image retrieval ready

### Demo 1 — Article similarity search (no filter)

You will begin with a pure similarity search over the restaurant article collection.

This experiment demonstrates:

- Baseline semantic retrieval behavior  
- Top-k nearest neighbor search  
- Retrieval without metadata constraints  

The results serve as a reference for later filtered retrieval experiments.



```python
# ================================
# Demo 1 — Article similarity search (no filter)
# ================================

q = "cozy restaurant with noodles and warm atmosphere"

ids, docs, metas, dists = retrieve_articles(q, k=5, where=None)
print_hits(ids, docs, metas, dists, title="Demo 1 — Article similarity search (no filter)")

print("✅ Demo 1 complete")

```

    
    === Demo 1 — Article similarity search (no filter) ===
    [1] id=rest_73 | cuisine=Hand-Pulled Noodle | location=San Gabriel | source=restaurant | distance=0.7496
    Restaurant: The Noodle Nest Cuisine: Hand-Pulled Noodle Location: San Gabriel
    [2] id=rest_132 | cuisine=Fusion Ramen | location=Sacramento | source=restaurant | distance=0.8238
    Restaurant: The Neon Noodle Cuisine: Fusion Ramen Location: Sacramento
    [3] id=rest_164 | cuisine=Japanese-Vietnamese | location=Anaheim | source=restaurant | distance=0.8319
    Restaurant: The Neon Noodle Cuisine: Japanese-Vietnamese Location: Anaheim
    [4] id=rest_150 | cuisine=Taiwanese | location=Irvine | source=restaurant | distance=0.8425
    Restaurant: The Silk Noodle Cuisine: Taiwanese Location: Irvine
    [5] id=rest_208 | cuisine=artisan pasta | location=Silver Lake | source=restaurant | distance=0.8915
    Restaurant: The Pasta Project Cuisine: artisan pasta Location: Silver Lake
    ✅ Demo 1 complete

### Demo 2 — Article similarity search + metadata filter

You will now introduce **metadata-constrained retrieval**, a key technique for improving precision in real-world systems. In addition to semantic similarity, we apply structured filters (for example, by location or cuisine). 

This hybrid approach allows the system to:

- Narrow the candidate search space  
- Enforce business or user constraints  
- Improve relevance under structured requirements  

You may experiment with different metadata values present in the dataset to observe how filtering affects recall and ranking.



```python
# ================================
# Demo 2 — Article similarity search + metadata filter
# ================================

q = "handmade pasta and romantic dinner"

# ---- metadata constraint (must exist in your dataset) ----
where_filter = {"location": "Pasadena"}  # adjust if needed

ids, docs, metas, dists = retrieve_articles(q, k=5, where=where_filter)

if len(ids) == 0:
    print("⚠️ No results found with current filter.")
else:
    print_hits(ids, docs, metas, dists, title="Demo 2 — Article similarity search + metadata filter")
    
print("✅ Demo 2 complete")

```

    
    === Demo 2 — Article similarity search + metadata filter ===
    [1] id=rest_6 | cuisine=small-plates | location=Pasadena | source=restaurant | distance=1.2173
    Restaurant: Velvet & Vine Cuisine: small-plates Location: Pasadena
    [2] id=rest_61 | cuisine=Plant-Based | location=Pasadena | source=restaurant | distance=1.3677
    Restaurant: Urban Harvest Cuisine: Plant-Based Location: Pasadena
    [3] id=rest_112 | cuisine= | location=Pasadena | source=restaurant | distance=1.3838
    Restaurant: The Green Gable Cuisine:  Location: Pasadena
    [4] id=rest_182 | cuisine=glamorous and theatrical Victorian excess | location=Pasadena | source=restaurant | distance=1.3872
    Restaurant: The Gilded Rose Cuisine: glamorous and theatrical Victorian excess Location: Pasadena
    [5] id=rest_160 | cuisine=Victorian | location=Pasadena | source=restaurant | distance=1.4137
    Restaurant: The Gilded Rose Cuisine: Victorian Location: Pasadena
    ✅ Demo 2 complete

### Demo 3 — Image similarity search (image→image)

Finally, you will perform visual similarity retrieval using a stored food image as the query.

This experiment demonstrates:

- CLIP-based image embeddings  
- Nearest neighbor search in visual space  
- Optional metadata constraints on images  

The query image can be changed by modifying the `QUERY_INDEX` parameter, enabling interactive exploration of the image collection.



```python
# ================================
# Demo 3 — Image similarity search (image→image)
# ================================

meta_all = image_db._collection.get(include=["metadatas"])["metadatas"]

QUERY_INDEX = 0  # ← change this (0 … N-1) to try different images

if QUERY_INDEX >= len(meta_all):
    raise ValueError("QUERY_INDEX out of range.")

query_img = meta_all[QUERY_INDEX]["image_path"]

print(f"Query image: {query_img}")
img = Image.open(query_img)
img.thumbnail((300, 300))
display(img)
# display(Image.open(query_img))

# TODO:
# 1. Create an optional metadata filter for recipe images
# 2. Retrieve the top-5 most similar images using query_img
# 3. Display the retrieved results with metadata (use title="Demo 3 — Image similarity search (image→image)" when printing results)

# your code here

print("✅ Demo 3 complete")
print("🎉 Similarity Retrieval with Metadata Filtering COMPLETE")

```

    Query image: recipe_images/synthetic_recipe_images/recipe1.png


    
![png](output_22_1.png)
    


    ✅ Demo 3 complete
    🎉 Similarity Retrieval with Metadata Filtering COMPLETE

### 📸 Screenshot Submission Requirement

At the end of this lab, take a screenshot of the final code cell and its output, and name it 
**M2L2_similarity_retrieval.jpg**.

Your screenshot must clearly show:

- A retrieval query (image)
- Displayed top-K retrieved restaurant results (with metadata)
- The final completion message:  
  **"Similarity Retrieval with Metadata Filtering COMPLETE"**

This screenshot will later serve as a component for the assessment.


## Conclusion

In this lab, you implemented similarity-based retrieval with metadata constraints over multimodal vector databases. You compared unconstrained and constrained retrieval for restaurant articles and executed a visual similarity search over the food image collection.

These retrieval capabilities form the core of production-grade multimodal RAG systems and prepare the foundation for multimodal fusion and re-ranking in the next lesson.


## Authors


[Zikai Dou](https://author.skills.network/instructors/zikai_dou)


Copyright © IBM Corporation. All rights reserved.

