from fastapi import FastAPI, HTTPException
from app.graph import app_graph # esto importa la cadena de agentes ya creada
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

app = FastAPI(
    title="Nombre de la app",
    description="Descripción",
    version="1.0.0"
)

class ChatInput(BaseModel):
    message: str
    thread_id: str

# endpoint para el chatbot
@app.post("/api/chat")
async def chat(user_input: ChatInput):
    config = {"configurable": {"thread_id": user_input.thread_id}}
    mensaje_nuevo = HumanMessage(content=user_input.message)
    resultado = app_graph.invoke({"mensajes": [mensaje_nuevo]}, config=config)
    return {"respuesta": resultado['respuesta_final']}

    # lo que tiramos aquí:
    # 1: sesión de chat
    # 2: metemos el mensaje del usuario
    # 3: generamos la respuesta
    # 4: la devolvemos


# este endpoint sirve para verificar de que el servidor funcione
@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "Versión 1.0.0"}


# arranque del servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)