okei entonces me inscribi en candidates 2026 en el reto de HRI/Vision sin saber nada de ROS 2 literalmente nada. el reto es construir un sistema que vea y escuche a personas, identifique quien habla y responda su pregunta usando su documento privado. suena cool pero es bastante complejo porque hay que fusionar tres señales (cara, voz, boca) y ninguna sola es suficiente para identificar a alguien.

al final el sistema funciona, levanta con un solo comando, tiene deteccion de caras con cajas y nombres en pantalla, microfono activo, transcripcion en tiempo real, RAG completamente local con Ollama y ChromaDB, fusion de identidad con tres estados de decision, y anti-spoofing basico. 

lo mas dificil de todo fue macOS. ROS 2 en mac es complicado y muchas cosas no funcionan igual que en linux. eso me quito mucho tiempo.
