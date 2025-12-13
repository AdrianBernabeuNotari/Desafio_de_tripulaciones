import os
from dotenv import load_dotenv

# Librerías de LangChain
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# [CIBER]: Configuración de Rutas
# Debe coincidir EXACTAMENTE con la ruta definida en scripts/ingest_data.py
DB_PATH = "./data/chroma_db"

# ==========================================
# 1. CONFIGURACIÓN DE CONEXIÓN
# ==========================================

def get_vectorstore():
    """
    Establece la conexión con la base de datos Chroma existente en disco.
    """
    # [CIBER]: Integridad de los datos.
    # Es CRÍTICO usar el mismo modelo de embeddings que se usó en la ingesta.
    # Si cambias el modelo aquí, los vectores no coincidirán y la búsqueda fallará (o devolverá basura).
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"⚠️ No se encuentra la base de datos en {DB_PATH}. Ejecuta ingest_data.py primero.")

    # Conectamos en modo lectura
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function
    )
    return vectorstore

# ==========================================
# 2. FUNCIÓN DE BÚSQUEDA (RETRIEVAL)
# ==========================================

def query_knowledge_base(query: str, k: int = 3) -> str:
    """
    Busca documentos relevantes en ChromaDB basados en la query.
    
    Args:
        query (str): La frase de búsqueda generada por el Agente 2.
        k (int): Número máximo de fragmentos a recuperar (Top-K).
        
    Returns:
        str: Un texto único concatenando la información encontrada.
    """
    try:
        db = get_vectorstore()
        
        # [CIBER]: Similarity Search con Score Threshold (Umbral).
        # En lugar de .similarity_search simple, usamos search_with_score.
        # Esto permite filtrar resultados que, aunque sean los "más cercanos",
        # sigan siendo irrelevantes (distancia muy alta).
        
        # Realizamos la búsqueda
        # k=3 significa que solo recuperamos los 3 trozos más relevantes.
        # Esto limita la "superficie de exposición" de los datos.
        results_with_scores = db.similarity_search_with_score(query, k=k)
        
        if not results_with_scores:
            return "No se encontró información relevante en los protocolos."

        # Procesar resultados
        context_parts = []
        for doc, score in results_with_scores:
            # [CIBER]: Umbral de relevancia.
            # En distancia Euclidiana/Coseno de Chroma, menor score = más similitud.
            # Un score > 1.5 (dependiendo del modelo) suele ser ruido.
            # Aquí podrías poner: if score > 1.0: continue
            
            # Limpieza del contenido (quitar saltos de línea excesivos)
            content = doc.page_content.replace("\n", " ")
            
            # Añadir metadatos (ej. página del PDF) para trazabilidad
            source = doc.metadata.get("source", "desconocido")
            page = doc.metadata.get("page", 0)
            
            context_parts.append(f"[Fuente: {source} (Pág {page})]: {content}")

        return "\n\n".join(context_parts)

    except Exception as e:
        print(f"Error en RAG: {e}")
        return "Error al consultar la base de datos de conocimiento."