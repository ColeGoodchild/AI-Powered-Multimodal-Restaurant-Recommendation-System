<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# **Multimodal Similarity Fusion and Retrieval Ranking**


Estimated time needed: **45** minutes


## Introduction

Earlier, you built the multimodal retrieval backbone by constructing vector indexes for restaurant articles and food images. You then operationalized these indexes by performing similarity-based retrieval with optional metadata filtering. At that stage, ranking was performed independently within each modality.

In this lab, you take the next step toward a production-grade multimodal retrieval system. Real-world RAG pipelines rarely rely on a single evidence source. Instead, they retrieve candidates from multiple modalities and **fuse** them into a unified ranking that better reflects overall relevance.

To enable this, the lab introduces **cross-modal score normalization and fusion**. Because similarity scores produced by different embedding models (for example, Sentence-Transformers versus CLIP) are not directly comparable, they must first be calibrated onto a common scale. You'll then apply a weighted fusion strategy to combine evidence from text and image retrieval into a single ranked list.

You'll design and implement a practical multimodal fusion pipeline that mirrors real production systems. This includes retrieving candidates from both modalities, normalizing their scores, applying weighted fusion, and optionally enforcing metadata constraints to support controllable, precision-oriented retrieval.

By the end of the lab, you will have a unified multimodal ranking pipeline capable of intelligently prioritizing heterogeneous evidence sources.


## Objectives

After completing this lab, you will be able to:

- Compute cosine-based similarity scores for both text and image retrieval pipelines  
- Normalize similarity scores to enable fair comparison across embedding spaces  
- Implement weighted multimodal fusion to produce a unified ranked result set  
- Apply metadata constraints to support controllable, constraint-aware reranking  
- Analyze how fusion weights and filtering strategies affect retrieval behavior
- Build a production-style multimodal retrieval workflow that integrates heterogeneous evidence sources


## Set up of the environment 

In this lab, you will produce a **unified ranked list** by combining evidence from both the article and image indexes. We begin by importing the core libraries for numerical computation, model inference, and vector database access.

All required packages were installed in Lesson 1, so this step verifies that the runtime environment is ready.



```python
# ================================
# Import environment
# (Dependencies were installed in Lesson 1)
# ================================

# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")

```

    ✅ Environment ready

## Verify vector databases

Before implementing multimodal fusion, you'll confirm that the Chroma databases created earlier are available and populated.

This validation step prevents downstream errors (for example, empty retrieval results) and ensures that this lab operates on the same persisted indexes used in earlier labs.



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

article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
image_db   = Chroma(collection_name="food_images",          persist_directory=DB_DIR)

n_articles = article_db._collection.count()
n_images   = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError(
        "One or more collections are empty. Please rerun Lesson 1 to rebuild the index."
    )

print(f"✅ Article vectors: {n_articles}")
print(f"✅ Image vectors:   {n_images}")
```

    ✅ Article vectors: 210
    ✅ Image vectors:   109

## Initialize embedding models

To retrieve across modalities, you need to embed queries into the **same vector spaces** used when building the indexes:

- **Text embeddings (384-d)** for article retrieval (Sentence-Transformers)  
- **CLIP embeddings (512-d)** for image retrieval  
  - Earlier, you focused on **image → image**  
  - This lab adds **text → image** retrieval using CLIP’s text encoder  

All embeddings are **L2-normalized**, so cosine similarity is stable and comparable across modalities.

Because different models produce scores on different scales, you will **normalize similarity scores** later before fusing results across modalities.



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


# ---- CLIP embedding model (512-d) for image + query text ----
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

@torch.no_grad()
def embed_query_clip_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**inputs)              # (1,512)
    feats = feats / feats.norm(dim=-1, keepdim=True)            # cosine-ready
    return feats[0].cpu().numpy().astype(np.float32)

print("✅ CLIP embedders ready")
```

    ✅ Text embedder ready✅ CLIP embedders ready

## Prepare utility functions

Chroma returns nested result structures (lists-of-lists) and **distance**-based scores. To support clean fusion logic, you will implement helper utilities that:

- Unwrap retrieval outputs  
- Convert cosine-style distances into similarity scores: `similarity = 1 - distance`
- Normalize similarity scores to `[0, 1]` for fusion  

These utilities ensure that cross-modal comparisons are meaningful and consistent.



```python
# ================================
# Utilities
# ================================

def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first query."""
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists

def _to_similarity(dists):
    """Convert 'smaller is better' distance to 'larger is better' similarity."""
    d = np.array(dists, dtype=np.float32)
    return 1.0 - d

def _minmax(x):
    """Min-max normalize to [0, 1] with safe handling for constant arrays."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)  # all equal -> treat as same confidence
    return (x - lo) / (hi - lo)

def print_hits(ids, docs, metas, scores, title: str, max_chars: int = 140):
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        score = float(scores[i]) if i < len(scores) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", ids[i]) if isinstance(meta, dict) else ids[i]
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | score={score:.4f}")
        print(f"{snippet}")
```

## Define Retrieval functions (text → articles, text → images)

Next, define two retrieval functions that accept the **same user query** but search **different vector indexes**. The query is encoded into:

- **MiniLM text space (384-d)** for text→article retrieval  
- **CLIP multimodal space (512-d)** for text→image retrieval  

Each embedding retrieves top-k candidates from its corresponding collection. To support downstream fusion, both functions return a **standardized output format** (IDs, documents, metadata, and similarity scores), enabling cross-modal normalization and unified ranking. Both functions optionally accept metadata filters via `where=...`, enabling **constraint-aware retrieval** (for example, restricting by `source`, `location`, or `cuisine`).



```python
# ================================
# Retrieval functions
# ================================

def retrieve_articles(query: str, k: int = 5, where: dict | None = None):
    q_vec = embed_texts([query])[0]  # 384-d
    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    sims = _to_similarity(dists)
    return ids, docs, metas, sims

def retrieve_images_by_text(query: str, k: int = 5, where: dict | None = None):
    q_vec = embed_query_clip_text(query)  # 512-d
    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    sims = _to_similarity(dists)
    return ids, docs, metas, sims

print("✅ Retrieval functions ready")
```

    ✅ Retrieval functions ready

## Implement multimodal fusion and reranking

You will now combine article and image results into a **single ranked list**. 

The fusion process:

1. Retrieves top-k candidates from each modality  
2. Normalizes similarity scores within each modality  
3. Computes a weighted fused score:
   $$
   s_\text{fused} = w_\text{text} \cdot \hat{s}_\text{text} + w_\text{img} \cdot \hat{s}_\text{img}
   $$
4. Produces a single unified ranked list  

This fusion step enables cross-modal comparison and ranking. By adjusting modality weights and metadata filters, you can control the relative influence of text and image evidence, mirroring the production multimodal RAG systems.



```python
# ================================
# Multimodal fusion
# ================================

def fuse_rank(
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict | None = None,
    where_img: dict | None = None,
    top_n: int = 5
):
    # Retrieve per modality
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(query, k=k_text, where=where_text)
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(query, k=k_img, where=where_img)

    # Normalize within modality
    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    # Build one mixed candidate list with fused scores
    rows = []
    for j in range(len(t_ids)):
        rows.append({
            "modality": "article",
            "id": t_metas[j].get("doc_id", t_ids[j]) if isinstance(t_metas[j], dict) else t_ids[j],
            "cuisine": t_metas[j].get("cuisine", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "location": t_metas[j].get("location", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "source": t_metas[j].get("source", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "text_score": float(t_norm[j]),
            "img_score": 0.0,
            "fused": float(w_text * t_norm[j]),
            "snippet": (t_docs[j] or "").replace("\n", " ").strip(),
        })

    for j in range(len(i_ids)):
        rows.append({
            "modality": "image",
            "id": i_metas[j].get("doc_id", i_ids[j]) if isinstance(i_metas[j], dict) else i_ids[j],
            "cuisine": i_metas[j].get("cuisine", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "location": i_metas[j].get("location", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "source": i_metas[j].get("source", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "text_score": 0.0,
            "img_score": float(i_norm[j]),
            "fused": float(w_img * i_norm[j]),
            "snippet": (i_docs[j] or "").replace("\n", " ").strip(),
        })

    # Sort by fused score (desc rerank)
    rows.sort(key=lambda r: r["fused"], reverse=True)
    
    # if top_n not specified, return full pool (k_text + k_img)
    if top_n is None:
        return rows

    top_n = max(0, min(int(top_n), len(rows)))
    return rows[:top_n]

def print_fused(rows, title: str, max_chars: int = 90):
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{idx}] {r['modality']} | id={r['id']} | cuisine={r['cuisine']} | "
            f"location={r['location']} | fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )
        print(snippet)
```

## Demo 1 — Multimodal fusion (no filters)

Begin with the baseline scenario. The query retrieves candidates from both modalities and fuses them using default weights.

- **Article database:** text → text similarity  
- **Image database:** text → image similarity  

This demo demonstrates the full multimodal pipeline without any metadata constraints. 



```python
# ================================
# Demo 1 — Multimodal fusion (no filters)
# ================================

q = "cozy noodles with warm atmosphere"

rows = fuse_rank(
    q,
    k_text=5,
    k_img=5,
    w_text=0.6,
    w_img=0.4,
    where_text=None,
    where_img=None,
    top_n=5
)

print_fused(rows, title="Demo 1 — Multimodal fusion (no filters)")
print("✅ Demo 1 complete")
```

    
    === Demo 1 — Multimodal fusion (no filters) ===
    [1] article | id=rest_73 | cuisine=Hand-Pulled Noodle | location=San Gabriel | fused=0.6000 (text=1.0000, img=0.0000)
    Restaurant: The Noodle Nest Cuisine: Hand-Pulled Noodle Location: San Gabriel
    [2] article | id=rest_164 | cuisine=Japanese-Vietnamese | location=Anaheim | fused=0.4063 (text=0.6771, img=0.0000)
    Restaurant: The Neon Noodle Cuisine: Japanese-Vietnamese Location: Anaheim
    [3] image | id=img_8 | cuisine=Middle Eastern | location=N/A | fused=0.4000 (text=0.0000, img=1.0000)
    Chickpea Falafel with Tahini
    [4] article | id=rest_150 | cuisine=Taiwanese | location=Irvine | fused=0.3730 (text=0.6216, img=0.0000)
    Restaurant: The Silk Noodle Cuisine: Taiwanese Location: Irvine
    [5] article | id=rest_132 | cuisine=Fusion Ramen | location=Sacramento | fused=0.3724 (text=0.6206, img=0.0000)
    Restaurant: The Neon Noodle Cuisine: Fusion Ramen Location: Sacramento
    ✅ Demo 1 complete

## Demo 2 — Multimodal fusion (metadata filters)

In production, you often want semantic relevance and constraints (e.g., “only Pasadena” or “only Italian”).

Here, you'll apply:
- A filter on the article database (for example, location)
- A filter on the image database (for example, source, cuisine)

If a filter is too strict, you may see fewer results. Observe how filtering affects the candidate pool and the final fused ranking.



```python
# ================================
# Demo 2 — Multimodal fusion (metadata filters)
# ================================

q = "handmade pasta and romantic dinner"

where_articles = {"location": "Pasadena"}   # change to any location present in your dataset
where_images   = {"source": "recipe_image"} # optional

rows = fuse_rank(
    q,
    k_text=5,
    k_img=5,
    w_text=0.6,
    w_img=0.4,
    where_text=where_articles,
    where_img=where_images,
    top_n=5
)

if len(rows) == 0:
    print("⚠️ No results found. Try relaxing filters (location/cuisine/source).")
else:
    print_fused(rows, title="Demo 2 — Multimodal fusion (metadata filters)")

print("✅ Demo 2 complete")
```

    
    === Demo 2 — Multimodal fusion (metadata filters) ===
    [1] article | id=rest_6 | cuisine=small-plates | location=Pasadena | fused=0.6000 (text=1.0000, img=0.0000)
    Restaurant: Velvet & Vine Cuisine: small-plates Location: Pasadena
    [2] image | id=img_83 | cuisine=American | location=N/A | fused=0.4000 (text=0.0000, img=1.0000)
    Salisbury Steak with Mushroom Gravy
    [3] image | id=img_16 | cuisine=Indian | location=N/A | fused=0.2109 (text=0.0000, img=0.5272)
    Falooda (Dessert Drink)
    [4] article | id=rest_61 | cuisine=Plant-Based | location=Pasadena | fused=0.1405 (text=0.2341, img=0.0000)
    Restaurant: Urban Harvest Cuisine: Plant-Based Location: Pasadena
    [5] article | id=rest_112 | cuisine= | location=Pasadena | fused=0.0915 (text=0.1526, img=0.0000)
    Restaurant: The Green Gable Cuisine:  Location: Pasadena
    ✅ Demo 2 complete

## Demo 3 — Weight tuning (reranking behavior)

Finally, you'll explore how fusion weights influence ranking behavior. By adjusting the fusion weights, you can control which modality dominates the final ranking:

- Higher `w_text` → more article-heavy ranking
- Higher `w_img` → more image-heavy ranking

Try changing weights and observe how the top results shift.



```python
# ================================
# Demo 3 — Weight tuning
# ================================

q = "fresh sushi and minimalist presentation"

# TODO:
# 1. Run fusion ranking with a text-heavy setting and print results (use title="Demo 3A — Text-heavy fusion (w_text=0.8, w_img=0.2)")
# 2. Run fusion ranking with an image-heavy setting and print results (use title="Demo 3B — Image-heavy fusion (w_text=0.3, w_img=0.7)")
# 3. Select 5 results from each modality, but only show the top 5 fused results

# your code here

print("✅ Demo 3 complete")
print("🎉 Multimodal Similarity Fusion and Retrieval Ranking COMPLETE")

```

    ✅ Demo 3 complete
    🎉 Multimodal Similarity Fusion and Retrieval Ranking COMPLETE

### 📸 Screenshot Submission Requirement

At the end of this lab, take a screenshot of the final code cell and its output, and name it  **M2L3_multimodal_fusion_results.jpg**.

Your screenshot must clearly show:
- Multimodal fusion results (ranked outputs)
- Fused ranking results for multiple weight configurations (for example, w_text=0.8 and w_text=0.3)
- The final completion message:  
  **"Multimodal Similarity Fusion and Retrieval Ranking COMPLETE"**

This screenshot will later serve as a component for the assessment.


## Conclusion

In this lab, you implemented a production-style multimodal retrieval pipeline that retrieves from text and image vector indexes, normalizes cross-modal similarity scores, and fuses results into a unified ranked list with optional metadata constraints.

This fusion-based reranking pattern is a core building block for high-precision multimodal RAG systems.


## Authors


[Zikai Dou](https://author.skills.network/instructors/zikai_dou)


Copyright © IBM Corporation. All rights reserved.

