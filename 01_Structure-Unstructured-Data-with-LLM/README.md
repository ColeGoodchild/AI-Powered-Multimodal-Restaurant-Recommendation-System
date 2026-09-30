The challenge
The organization has acquired a large, messy dataset spanning multiple modalities:

Restaurant descriptions and user reviews (raw TXT)
Food recipes with structured metadata and images (JSON + JPEG)
User restaurant visit histories (JSON with URLs)
However, this data is not immediately usable:

Unstructured text lacks a consistent schema, making search and retrieval inefficient.
Images and URLs are opaque to traditional data pipelines and must be transformed into searchable, quantitative representations.
The data spans multiple modalities with no unified representation.
To build a high-quality, explainable recommendation engine, your first task is to transform this heterogeneous, multimodal data into a structured knowledge base. This includes extracting semantic signals from text, generating representations for images and URLs, and organizing everything into a coherent, query-friendly JSON format.
