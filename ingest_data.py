# Este código busca los pdfs de información que le queremos meter al chatbot
# y los divide en trozos, para poder vectorizarlos (pasarlos a números) y guardarlos.
# Para nuestro proyecto solo lo ejecutamos una vez
# cuando ya tengamos el pdf, la bd y los vectores ya no nos hace falta tirarlo.

import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "./data/raw"
DB_PATH = "./data/chroma_db"

def main():

    # Carga
    print("Iniciando proceso de ingesta de datos...")

    if not os.path.exists(DATA_PATH):
        print(f"Error: La carpeta {DATA_PATH} no existe.")
        return

    print(f"Cargando PDFs desde {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("No se encontraron documentos. Abortando.")
        return

    print(f"Se han cargado {len(documents)} documentos.")

    # División
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Tamaño del fragmento (caracteres)
        chunk_overlap=200,    # Solapamiento para no cortar frases a medias
        add_start_index=True, # Mantiene metadatos de posición
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(chunks)} fragmentos.")

    # Embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Creando/Actualizando Base de Datos Chroma (esto puede tardar)...")
    
    # Borrar DB anterior
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) 

    # Crear la DB persistente
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"¡Éxito! Base de datos vectorial guardada en: {DB_PATH}")

if __name__ == "__main__":
    main()