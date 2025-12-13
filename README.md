# Repositorio temporal del desafío de tripulaciones para poder compartir código entre Ciber y Data

Lo que hay por ahora para el chatbot con LLM: 
  
- **main.py**: solo tiene el endpoint para el chatbot, pero para que se vea un poco  
- **cifrado_nuevo.py**: el archivo para cifrar que nos habíais enviado (el primero con endpoint ya me diréis si hay que meterlo también)  
- **ingest_data.py**: este archivo lo que hace es coger un documento de texto que hay en data/raw para darle info al chatbot (convierte la info a número para buscarla mejor luego)  
- **data/raw**: aquí está el documento de texto que le da explicaciones sobre el acoso escolar al bot  
- **data/chroma_db**: aquí hay una base de datos vectorial, sirve para decirle al bot que información tiene que buscar  
- **app/graph.py**: este es el sistema multiagente, los ordena (lo explico abajo)  
- **app/agentes**: aquí están los agentes, cada uno tiene su rol (más adelante los modificamos para que funcionen mejor)  
- **app/rag_utils.py**: este archivo convierte los mensajes del usuario a números, para poder buscar la información en el rag comparándo números  
- **.env**: aquí meteremos la API key de OpenAI cuando "activemos" el chatbot

Esta carpeta está en desarollo, van a cambiar cosas, pero en principio el código se debería de quedar bastante parecido (?), salvo para el main.py que habrá que rellenarlo entero.  
Este repo por ahora es para ver únicamente la parte del chatbot.  
  
## El sistema multiagentes básicamente lo que hace básicamente es crear tres agentes encargados de:  

### Reconocimiento: 

Detecta si está hablando con alguien a secas, con una víctima de acoso escolar, un testigo o un agresor.  
Le dice al siguiente agente con qué tipo de persona está hablando para darle contexto.  

### Búsqueda:

Ahora que sabe con quién habla, este agente busca en la base de datos (vectorial) la información más adecuada para la situación de cada usuario.

### Respuesta:

Este agente es el que se encarga de responder al usuario. Básicamente coge la información que ha encontrado el segundo agente con ayuda del contexto del primero, para generar una respueste adapatada al usuario.  
  
#### Algunas cosas que tendremos en cuenta:
  
- que el chatbot no minimice los problemas de la gente (que no te ignore vamos)
- que nunca incite a la violencia
- que tenga en cuenta que el agresor, aunque sea de forma distinta, también necesita un apoyo especializado, puede ser la víctima de otra historia