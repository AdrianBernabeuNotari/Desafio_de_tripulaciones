from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from app.agents import profiler_chain, query_chain, responder_chain, PerfilUsuario
from app.rag_utils import query_knowledge_base


# Esto es un diccionario con la información que los agentes se pasan entre ellos
class GraphState(TypedDict):
    mensajes: Annotated[list, add_messages]   # Historial de mensajes
    perfil: dict                              # Perfil del usuario (Agente 1)
    search_query: str                         # Info a buscar en el RAG (Agente 2)
    contexto: str                             # Info recuperada (Herramienta de búsqueda)
    respuesta_final: str                      # Respuesta final (Agente 3)


# Aquí definimos los nodos
# Básicamente son los agentes y la herramienta de búsqueda en el rag

# Agente 1: analiza el mensaje del usuario y define el perfil
def nodo_profiler(state: GraphState):

    print("Agente 1: perfilador, borra este mensaje antes de tirar la versión final!")
    mensaje = state["mensajes"][-1].content
    resultado: PerfilUsuario = profiler_chain.invoke({"mensaje": mensaje})
    return {"perfil": resultado.dict()}


# Agente 2: prepara la query para ir a buscar la info en el RAG
def nodo_generador_query(state: GraphState):
    
    print("Agente 2: generador de query, borra este mensaje antes de tirar la versión final!")
    perfil = state["perfil"]
    resultado_query = query_chain.invoke({
        "rol": perfil["rol"], 
        "resumen": perfil["resumen_situacion"]
    })
    return {"search_query": resultado_query.content}


# Herramienta RAG: Ejecuta la búsqueda en ChromaDB
def nodo_buscador(state: GraphState):
    
    print(" Herrameinta: búsqueda en chroma, borra este mensaje antes de tirar la versión final!")
    query = state["search_query"]
    documentos = query_knowledge_base(query)
    return {"contexto": documentos}


# Agente 3: Escribe la respuesta final
def nodo_redactor(state: GraphState):
    
    print("Agente 3: redactor, borra este mensaje antes de tirar la versión final!")
    ultimo_mensaje_texto = state["mensajes"][-1].content
    respuesta_chain = responder_chain.invoke({
        "history": state["mensajes"],
        "rol": state["perfil"]["rol"],
        "riesgo": state["perfil"]["nivel_riesgo"],
        "contexto": state["contexto"],
        "mensaje": ultimo_mensaje_texto
    })
    return {"respuesta_final": respuesta_chain.content, "mensajes": [respuesta_chain]}


# Aquí construimos el grafo
# es decir, el flujo de trabajo

workflow = StateGraph(GraphState)

# Añadimos los nodos
workflow.add_node("profiler", nodo_profiler)
workflow.add_node("query_gen", nodo_generador_query)
workflow.add_node("rag_tool", nodo_buscador)
workflow.add_node("redactor", nodo_redactor)

# Definimos el flujo
# Flujo lineal simple: Inicio -> Profiler -> QueryGen -> RAG -> Redactor -> Fin
workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "query_gen")
workflow.add_edge("query_gen", "rag_tool")
workflow.add_edge("rag_tool", "redactor")
workflow.add_edge("redactor", END)

# Creamos la memoria
memory = MemorySaver()

# Compilamos la aplicación
app_graph = workflow.compile(checkpointer=memory) # Actualizamos la memoria