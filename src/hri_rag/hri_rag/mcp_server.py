"""
Servidor MCP (Model Context Protocol) - expone RAG como herramienta HTTP.

Corre con: python -m hri_rag.mcp_server
O desde ROS 2: ros2 run hri_rag mcp_server

Usa: curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"identity_name":"Gali","identity_confidence":0.9,"question":"¿Cuál es mi proyecto?"}'
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import os
import logging

from hri_rag.document_ingestion import search_document


OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'llama3.2:1b'
CONFIDENCE_THRESHOLD = 0.55  # umbral mínimo para responder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mcp_server')

app = FastAPI(title='HRI RAG MCP Server')


class QueryRequest(BaseModel):
    identity_name: str
    identity_confidence: float
    question: str


class QueryResponse(BaseModel):
    success: bool
    answer: str


@app.post('/rag/query', response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """Endpoint RAG con validación de identidad."""
    logger.info(
        f'Consulta: {request.identity_name} '
        f'(confianza: {request.identity_confidence:.0%}) '
        f'pregunta: "{request.question}"'
    )

    # Validación de identidad - REGLA CRÍTICA
    if request.identity_confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            f'Confianza insuficiente ({request.identity_confidence:.0%} < {CONFIDENCE_THRESHOLD:.0%})'
        )
        return QueryResponse(
            success=False,
            answer=f'No puedo verificar tu identidad (confianza: {request.identity_confidence:.0%}). '
                   f'Necesito al menos {CONFIDENCE_THRESHOLD:.0%}.'
        )

    # Búsqueda en ChromaDB
    relevant_chunks = search_document(request.identity_name, request.question, top_k=3)

    if not relevant_chunks:
        return QueryResponse(
            success=False,
            answer=f'No encontré información sobre "{request.question}" en tu documento.'
        )

    # generacion de respuesta con Ollama
    context = '\n'.join(relevant_chunks)
    prompt = f"""Eres un asistente privado. Responde UNICAMENTE usando la informacion del documento proporcionado.
REGLAS ESTRICTAS:
- Si la respuesta NO esta en el documento, responde exactamente: "No tengo esa informacion en el documento."
- NO inventes, NO uses conocimiento externo, NO supongas.
- Responde en español, de forma concisa y directa.

DOCUMENTO DE {request.identity_name.upper()}:
{context}

PREGUNTA: {request.question}

RESPUESTA (solo con informacion del documento):"""

    try:
        response_text = await generate_response(prompt)
        return QueryResponse(success=True, answer=response_text)
    except Exception as e:
        logger.error(f'Error generando respuesta: {e}')
        return QueryResponse(
            success=False,
            answer='Error al generar respuesta. Intenta de nuevo.'
        )


async def generate_response(prompt: str) -> str:
    """Llama a Ollama y obtiene la respuesta."""
    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'temperature': 0.5
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get('response', '').strip()
    except Exception as e:
        logger.error(f'Error llamando a Ollama: {e}')
        raise


@app.get('/health')
async def health():
    """Health check."""
    return {'status': 'ok'}


def main():
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level='info'
    )


if __name__ == '__main__':
    main()
