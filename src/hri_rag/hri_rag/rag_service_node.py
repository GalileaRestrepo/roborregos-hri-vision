import rclpy
from rclpy.node import Node
from hri_interfaces.srv import RAGQuery
from hri_rag.document_ingestion import search_document
import requests
import os


OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'llama3.2:1b'
CHROMA_DB_PATH = os.path.expanduser('~/ros2_hri_ws/data/chroma_db')


class RagServiceNode(Node):
    def __init__(self):
        super().__init__('rag_service_node')

        self.service = self.create_service(
            RAGQuery,
            '/rag/query',
            self.handle_query
        )

        self.get_logger().info('Servicio RAG listo en /rag/query')

    def handle_query(self, request, response):
        """Maneja una consulta RAG: búsqueda + generación de respuesta."""
        person_name = request.identity_name
        question = request.question
        confidence = request.identity_confidence

        self.get_logger().info(
            f'Consulta RAG: {person_name} ({confidence:.0%}) pregunta: "{question}"'
        )

        # busca chunks relevantes en el documento de esa persona
        relevant_chunks = search_document(person_name, question, top_k=2)

        # umbral mínimo de confianza para ejecutar RAG
        if confidence < 0.45:
            response.authorized    = False
            response.denial_reason = f'Confianza insuficiente ({confidence:.2f}) para acceder al documento.'
            return response

        if not relevant_chunks:
            response.authorized    = True
            response.answer        = f'No encontré información relevante en tu documento sobre "{question}".'
            response.source_chunks = []
            return response

        # prepara el contexto para ollama
        context = '\n'.join(relevant_chunks)
        prompt = f"""Eres un asistente que responde preguntas basándote SOLO en el siguiente documento.
Responde de forma concisa y natural en español. Si la respuesta no está en el documento, di que no tienes esa información.

Documento:
{context}

Pregunta: {question}

Respuesta:"""

        try:
            response_text = self.generate_response(prompt)
            response.authorized    = True
            response.answer        = response_text
            response.source_chunks = relevant_chunks
            self.get_logger().info(f'Respuesta generada para {person_name}')
        except Exception as e:
            self.get_logger().error(f'Error generando respuesta: {e}')
            response.authorized    = False
            response.denial_reason = 'Error al generar respuesta. Intenta de nuevo.'

        return response

    def generate_response(self, prompt):
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
            self.get_logger().error(f'Error llamando a Ollama: {e}')
            raise


def main(args=None):
    rclpy.init(args=args)
    node = RagServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
