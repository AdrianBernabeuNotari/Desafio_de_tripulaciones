from fastapi import FastAPI
from app.graph import app_graph # esto importa la cadena de agentes ya creada

app = FastAPI(
    title="Nombre de la app",
    description="Descripción",
    version="1.0.0"
)

# endpoint para el chatbot
@app.post("/api/chat")
async def chat(mensaje: str):
    inputs = {"mensaje": mensaje} # 1
    resultado = app_graph.invoke(inputs) # 2
    return {"respuesta": resultado['final_response']} # 3

    # lo que tiramos aquí:
    # 1: metemos el mensaje del usuario
    # 2: generamos la respuesta
    # 3: la devolvemos


# este endpoint sirve para verificar de que el servidor funcione
@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "Versión 1.0.0"}