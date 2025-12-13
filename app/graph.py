from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# Importamos nuestros módulos anteriores
from app.agents import profiler_chain, query_chain, responder_chain, PerfilUsuario
from app.rag_utils import query_knowledge_base

# ==========================================
# 1. DEFINICIÓN DEL ESTADO (LA MEMORIA)
# ==========================================
# Este diccionario es lo que se pasan los agentes entre sí.
# Al principio solo tiene 'mensaje_entrada'. Al final, lo tendrá todo.

class GraphState(TypedDict):
    mensaje_entrada: str       # Lo que escribe el usuario
    perfil: dict               # Datos del usuario (Víctima/Testigo/Riesgo) - Salida del Agente 1
    search_query: str          # Lo que buscaremos en Chroma - Salida del Agente 2
    contexto: str              # Los documentos recuperados - Salida de rag_utils
    respuesta_final: str       # Lo que leerá el usuario - Salida del Agente 3

# ==========================================
# 2. DEFINICIÓN DE LOS NODOS (LAS ESTACIONES DE TRABAJO)
# ==========================================

def nodo_profiler(state: GraphState):
    """
    Agente 1: Analiza el mensaje y define el perfil.
    """
    print("--- 🕵️‍♀️ NODO 1: PROFILER (Analizando Riesgo) ---")
    mensaje = state["mensaje_entrada"]
    
    # Ejecutamos la cadena del profiler (definida en agents.py)
    resultado: PerfilUsuario = profiler_chain.invoke({"mensaje": mensaje})
    
    # Convertimos el objeto Pydantic a diccionario para guardarlo en el estado
    return {"perfil": resultado.dict()}

def nodo_generador_query(state: GraphState):
    """
    Agente 2a: Traduce el problema a términos de búsqueda.
    """
    print("--- 🧠 NODO 2: QUERY GEN (Preparando Búsqueda) ---")
    perfil = state["perfil"]
    
    # El LLM genera la query basada en el resumen y el rol
    resultado_query = query_chain.invoke({
        "rol": perfil["rol"], 
        "resumen": perfil["resumen_situacion"]
    })
    
    # resultado_query es un objeto AIMessage, extraemos el contenido (.content)
    return {"search_query": resultado_query.content}

def nodo_buscador(state: GraphState):
    """
    Herramienta RAG: Ejecuta la búsqueda en ChromaDB.
    (Este nodo no usa LLM, es pura función de Python)
    """
    print("--- 📚 NODO 3: RAG (Consultando ChromaDB) ---")
    query = state["search_query"]
    
    # Usamos nuestra función de rag_utils.py
    documentos = query_knowledge_base(query)
    
    return {"contexto": documentos}

def nodo_redactor(state: GraphState):
    """
    Agente 3: Escribe la respuesta final.
    """
    print("--- ✍️ NODO 4: REDACTOR (Generando Respuesta) ---")
    mensaje = state["mensaje_entrada"]
    perfil = state["perfil"]
    contexto = state["contexto"]
    
    # Invocamos al orientador con toda la información acumulada
    respuesta = responder_chain.invoke({
        "mensaje": mensaje,
        "rol": perfil["rol"],
        "contexto": contexto
    })
    
    return {"respuesta_final": respuesta.content}

# ==========================================
# 3. CONSTRUCCIÓN DEL GRAFO (EL TABLERO)
# ==========================================

workflow = StateGraph(GraphState)

# A. Añadimos los nodos
workflow.add_node("profiler", nodo_profiler)
workflow.add_node("query_gen", nodo_generador_query)
workflow.add_node("rag_tool", nodo_buscador)
workflow.add_node("redactor", nodo_redactor)

# B. Definimos el flujo (Las flechas)
# Flujo lineal simple: Inicio -> Profiler -> QueryGen -> RAG -> Redactor -> Fin
workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "query_gen")
workflow.add_edge("query_gen", "rag_tool")
workflow.add_edge("rag_tool", "redactor")
workflow.add_edge("redactor", END)

# C. Compilamos la aplicación
app_graph = workflow.compile()