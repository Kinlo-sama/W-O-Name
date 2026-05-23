# W-O-Name
Repositorio para templates de tareas en vision por computadora 

La versión de python a utilizar para cada una de las tareas es: 3.11  
La estructura de las carpetas es la siguiente:
- Core: Funciones principales como 
    - Engine: Procesos de entrenamiento
    - Graph: Representación de modelos
    - Pluggins: Modulos externos y extras
    - Registro de modelos 
- Vision: Entran tareas por vision por computadora propuestas
-  Architectures: Backbone, Neck, Decoders, Heads
- Losses: Funciones de perdida
- Evaluators: Metricas de evaluación
- Layers: Capas primitivas e.g: DobConv2D, ConvTrans, ConvDil, etc.

Los formatos propuestos para las configuraciones de cada tarea son los siguientes:
- YAML
- TOML
- JSON
- Python
- CLI
- web UI


## **Metas**

- Implementación **lazy registration**