semana 3 fue la buena. hice todo lo que debi de haber hecho en semana 1, semana 2, y ademas lo de semana 3. fue bastante intenso pero si sali adelante.

arrange con la instalacion, miniconda, mamba, el ambiente de ROS 2 con robostack. ya ahi empezaron los problemas, cmake 4 no es compatible con ROS 2 Humble entonces todo tronaba al compilar. me toco investigar bastante hasta encontrar que habia que pasarle los compiladores manualmente.

defini todas las interfaces propias (.msg, .srv, .action), hice los nodos de vision (camera_node, face_detection_node, face_tracker_node con IoU), el audio (VAD, transcripcion, identidad de voz), el RAG con ChromaDB y Ollama completamente local, y la fusion con los tres estados de decision.

el problema mas frustrante fue macOS. los nodos de ROS 2 lanzados como subprocesos no pueden abrir ventanas ni acceder a camara ni microfono. eso me quito mucho tiempo hasta encontrar que la solucion era correr todo en un solo proceso con MultiThreadedExecutor.

probe primero Resemblyzer para identidad de voz pero daba similitudes muy bajas, lo cambie por ECAPA-TDNN de SpeechBrain. tambien en algun momento el ambiente de conda se rompio completo porque resemblyzer instalo un paquete que sobreescribia el modulo typing de Python y rclpy dejaba de funcionar.
