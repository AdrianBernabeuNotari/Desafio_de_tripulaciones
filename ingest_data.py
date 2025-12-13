import os
import shutil
#from dotenv import load_dotenv

# Importaciones de LangChain
# Nota: Requiere instalar langchain-community, langchain-openai, langchain-chroma pypdf
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Cargar variables de entorno (API KEYs)
# [CIBER]: Esto evita hardcodear credenciales en el código.
# El archivo .env debe estar en el .gitignore.
#load_dotenv()

# Configuración de Rutas
# [CIBER]: Validar que estas rutas tengan los permisos de escritura/lectura correctos en el SO.
DATA_PATH = "./data/raw"
DB_PATH = "./data/chroma_db"

def main():
    print("🚀 Iniciando proceso de ingesta de datos...")

    # ---------------------------------------------------------
    # PASO 1: CARGAR DOCUMENTOS
    # ---------------------------------------------------------
    # [CIBER]: Punto crítico. Aquí se leen archivos del disco.
    # Riesgo: "Malicious File Upload". Si alguien deja un PDF con exploit en /raw,
    # el loader intentará procesarlo. Idealmente, Ciber debería escanear /raw antes.
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: La carpeta {DATA_PATH} no existe.")
        return

    print(f"📂 Cargando PDFs desde {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("⚠️ No se encontraron documentos. Abortando.")
        return

    print(f"✅ Se han cargado {len(documents)} documentos.")

    # ---------------------------------------------------------
    # PASO 2: DIVIDIR TEXTO (CHUNKING)
    # ---------------------------------------------------------
    # Dividimos el texto en trozos pequeños para que quepan en la ventana de contexto del LLM.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Tamaño del fragmento (caracteres)
        chunk_overlap=200,    # Solapamiento para no cortar frases a medias
        add_start_index=True, # Mantiene metadatos de posición
    )
    chunks = text_splitter.split_documents(documents)
    print(f"scissors: Documentos divididos en {len(chunks)} fragmentos (chunks).")

    # ---------------------------------------------------------
    # PASO 3: EMBEDDING Y ALMACENAMIENTO VECTORIAL
    # ---------------------------------------------------------
    # [CIBER]: Aquí los datos salen hacia una API externa (OpenAI) para ser vectorizados.
    # Si la privacidad es crítica (ej. datos de menores), se debe cambiar OpenAIEmbeddings
    # por HuggingFaceEmbeddings (modelo local que corre en tu máquina).
    
    # Opción A: Modelo en la Nube (OpenAI)
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Opción B: Modelo Local (Más privado, requiere más RAM)
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Creando/Actualizando Base de Datos Chroma (esto puede tardar)...")
    
    # [CIBER]: Chroma guarda los datos en texto plano y vectores en DB_PATH.
    # No hay encriptación "At Rest" por defecto en la versión open source de Chroma.
    # Si roban la carpeta 'chroma_db', roban todo el conocimiento del bot.
    
    # Borrar DB anterior para empezar de cero (Opcional, útil en desarrollo)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) 

    # Crear la DB persistente
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"🎉 ¡Éxito! Base de datos vectorial guardada en: {DB_PATH}")

if __name__ == "__main__":
    main()