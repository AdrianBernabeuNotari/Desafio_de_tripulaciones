from fastapi import FastAPI
from app.graph import app_graph # Importas tu grafo ya compilado

app = FastAPI()

@app.post("/chat")
async def chat(mensaje: str):
    # Invocas al grafo
    inputs = {"mensaje": mensaje}
    resultado = app_graph.invoke(inputs)
    return {"respuesta": resultado['final_response']}