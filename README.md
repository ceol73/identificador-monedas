# Identificaión y conteo de monedas con CV

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![NumPy](https://img.shields.io/badge/Numpy-Calculation-red)

Este proyecto es una herramienta de visión artificial desarrollada en Python que automatiza el proceso de detección, clasificación y suma de monedas (denominaciones de $1, $2, $5 y $10). 

A diferencia de los detectores tradicionales basados en tamaños fijos, este proyecto utiliza **algoritmos de agrupamiento (K-Means)** y **círculos de control** para adaptarse dinámicamente a la escala de la imagen, haciéndolo robusto ante cambios en la distancia de la cámara.

## Características Principales

- **Adaptabilidad de Escala:** No requiere calibración manual de píxeles. Utiliza referencias en la imagen para entender qué tamaño tiene cada moneda.
- **Clasificación Inteligente:** Implementa el algoritmo **K-Means Clustering** para agrupar las monedas en 4 categorías de tamaño automáticamente.
- **Preprocesamiento Robusto:** Utiliza operaciones morfológicas (dilatación y erosión) para eliminar ruido y rellenar imperfecciones en los objetos detectados.
- **Conteo Preciso:** Distingue entre las monedas a contar y los círculos de referencia.

## Requisitos del Sistema

Para ejecutar este proyecto, necesitas tener instalado **Python 3.x** y las siguientes bibliotecas:

* **OpenCV** (`opencv-python`): Para el procesamiento de imágenes.
* **NumPy** (`numpy`): Para operaciones matemáticas y manejo de matrices.

### Instalación

```bash
pip install opencv-python numpy
```

## Uso

1. **Preparación de la imagen:**
   * Utilizar un fondo negro (o muy oscuro) para facilitar la segmentación.
   * Colocar las monedas que se desean contar.
   * **Importante:** Se debe incluir 4 "círculos de control" (uno del tamaño para cada tipo de moneda) en la imagen.
   * No debe haber monedas ni círculos tocándose entre sí ni debe haber unos por encima de otros.

2. **Ejecución:** modifca la variable `ruta_imagen` en el script principal o pasa la ruta como argumento (si se implementa CLI).
```bash
python monedas.py
```

3. **Resultado:** El script abrirá una ventana mostrando la imagen procesada con las etiquetas de valor y mostrará en la consola el desglose total del dinero.

## Explicación técnia del algoritmo

El flujo de procesamiento sigue los siguientes pasos:

1. **Preprocesamiento**

La imagen se convierte a escala de grises y se aplica una binarización por umbral. Se seleccionan los píxeles con un valor > 100 para separar los objetos (monedas) del fondo negro.

```python
_, binaria = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY)
```

2. **Operaciones Morfológicas**

Para asegurar que las monedas sean detectadas como objetos sólidos y eliminar ruido (puntos blancos pequeños), se aplican transformaciones morfológicas:
- Dilatación: Expande las áreas blancas para rellenar huecos dentro de las monedas.
- Erosión: Contrae los bordes para separar objetos levemente unidos y eliminar falsos positivos pequeños.

3. **Detección y clasificación (K-means)**

En lugar de definir "una moneda de $10 mide 500px", el sistema recolecta las áreas de todos los objetos detectados y aplica K-Means con `K=4`.
- El algoritmo encuentra 4 centroides (tamaños promedio).
- Se asume que el grupo de menor tamaño es $1 y el mayor es $10.
- Finalmente, se resta 1 unidad a cada conteo para excluir los círculos de control.

## Estructura del proyecto

```plaintext
identificador-monedas/
│
├── monedas.py              # Script principal con la lógica de visión
├── README.md               # Documentación del proyecto
└── test-img/               # Carpeta con imágenes de ejemplo
    ├── test_01.png
    └── test_02.png
```
