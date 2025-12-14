from fastapi import FastAPI
from app.graph import app_graph

app = FastAPI(
    title="Nombre de la app",
    description="Descripción",
    version="1.0.0"
)

@app.post("/api/chat")
async def chat(mensaje: str):
    inputs = {"mensaje": mensaje}
    resultado = app_graph.invoke(inputs)
    return {"respuesta": resultado['final_response']}


@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "Versión 1.0.0"}