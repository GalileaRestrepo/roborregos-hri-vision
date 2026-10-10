"""
Ingesta de documentos - convierte PDF/TXT en embeddings y los guarda en ChromaDB.

Uso desde script:
    python -c "from hri_rag.document_ingestion import ingest_document; \
        ingest_document('Gali', 'docs/mi_proyecto.pdf')"
"""
import os
import re
import fitz # PyMuPDF para leer PDFs
import chromadb
import requests
import json
from pathlib import Path


OLLAMA_ENDPOINT = 'http://localhost:11434/api/embed'
OLLAMA_MODEL = 'nomic-embed-text'
CHUNK_SIZE = 500 # caracteres por chunk
CHUNK_OVERLAP = 100 # solapamiento para contexto
CHROMA_DB_PATH = os.path.expanduser('~/ros2_hri_ws/data/chroma_db')


def read_document(file_path):
    """Lee PDF o TXT y devuelve el texto."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f'Archivo no encontrado: {file_path}')

    if file_path.suffix.lower() == '.pdf':
        doc = fitz.open(file_path)
        text = ''
        for page in doc:
            text += page.get_text()
        return text
    elif file_path.suffix.lower() in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f'Formato no soportado: {file_path.suffix}. Usa PDF o TXT.')


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide el texto en chunks con solapamiento."""
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def get_embedding(text):
    """Genera embedding usando Ollama nomic-embed-text."""
    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={'model': OLLAMA_MODEL, 'input': text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get('embeddings', [[]])[0]
    except Exception as e:
        print(f'Error al generar embedding: {e}')
        return None


def ingest_document(person_name, file_path):
    """Ingesta un documento para una persona."""
    print(f'Leyendo {file_path}...')
    text = read_document(file_path)
    print(f'  Total: {len(text)} caracteres')

    print(f'Chunking...')
    chunks = chunk_text(text)
    print(f'  Total: {len(chunks)} chunks')

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=f'person_{person_name}',
        metadata={'hnsw:space': 'cosine'}
    )

    print(f'Generando embeddings para {person_name}...')
    for i, chunk in enumerate(chunks):
        if not chunk or len(chunk) < 10:
            continue

        embedding = get_embedding(chunk)
        if embedding is None:
            print(f'  Chunk {i+1}/{len(chunks)}: FALLÓ')
            continue

        collection.add(
            ids=[f'{person_name}_{i}'],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{'person': person_name, 'chunk_index': i}]
        )

        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(chunks)} chunks procesados...')

    print(f'✓ Documento de {person_name} almacenado en ChromaDB')
    print(f'  Colección: person_{person_name}')
    print(f'  Chunks: {len(chunks)}')


def search_document(person_name, query, top_k=3):
    """Busca en el documento de una persona."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name=f'person_{person_name}')
    except Exception as e:
        print(f'Colección para {person_name} no encontrada: {e}')
        return []

    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results.get('documents', [[]])[0] if results else []


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Uso: python document_ingestion.py <nombre_persona> <ruta_documento>')
        print('Ejemplo: python document_ingestion.py Gali ~/docs/mi_proyecto.pdf')
        sys.exit(1)

    person_name = sys.argv[1]
    file_path = sys.argv[2]

    ingest_document(person_name, file_path)
