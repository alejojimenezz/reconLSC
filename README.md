# Reconocimiento de Lenguaje de Señas Colombiano (LSC)

## Lenguaje

### [Python](https://www.python.org/downloads/)

Versión 3.12.10.

## Librerias

### [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419)

Paquete de bibliotecas y herramientas de python enfocada en tecnicas de inteligencia artificial y aprendizaje automático. Usada, en este proyecto para la detección de puntos de referencia de la mano.

<img width="2146" height="744" alt="image" src="https://github.com/user-attachments/assets/bbd50d74-1d2e-46ee-a301-d9f8918867dd" />

### [OpenCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) (cv2)

Open Source Computer Vision. Libreria de python aprovechada para capturar el video de la camara web, generando la interfaz gráfica básica del programa.

## Archivos

### Alfabeto - [alphabet.py](/alphabet.py)

1. Modulo de definición del mapa de la mano siguiendo la documentación de [Mediapipe](#Mediapipe).
2. Funciones de identificación de cada gesto correspondiente al abecedario del lenguaje de señas colombiano ([LSC](https://educativo.insor.gov.co/catdiccionario/alfabeto/)).
3. Diccionario de letras para el llamado desde [main.py](/main.py)

### Prueba de camara - [camTest.py](/camTest.py)

Primera prueba del funcionamiento del programa con llamado en un solo archivo.

### Prueba de libreria - [pipLibTest.py](/pipLibTest.py)

Archivo para pruebas de funcionamiento de requisitos de las librerias usadas.

### Prueba de nueva letra - [newLetterTest.py](/newLetterTest.py)

Archivo para pruebas de letras ya incluidas, o aun sin incluir, para mejorar su reconocimiento, sensibilidad, o definición del gesto para hacerlo más acertado.

### Principal - [main.py](/main.py)

Archivo ejecutable donde se llama el [alfabeto](#Alfabeto) y se forma la ventana de funcionamiento del programa de reconocimiento de LSC.
