import cv2
import numpy as np

def contar_monedas_con_referencia(ruta_imagen):
    # 1. Cargar imagen y preprocesamiento
    imagen_original = cv2.imread(ruta_imagen)
    if imagen_original is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return

    gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

    # 2. Binarización
    # Convertimos a binario excluyendo el fondo (valores > 100 son objetos)
    _, binaria = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY)

    # 3. Operaciones Morfológicas (Dilatación y Erosión)
    # Esto rellena los hoyos y asegura que el área detectada sea sólida
    kernel = np.ones((5, 5), np.uint8)
    dilatada = cv2.dilate(binaria, kernel, iterations=2)
    erosionada = cv2.erode(dilatada, kernel, iterations=1)

    # 4. Encontrar contornos
    contornos, _ = cv2.findContours(erosionada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtramos contornos muy pequeños (ruido) que no sean monedas ni referencias
    # Ajusta 'area_minima' si detectas puntos basura
    area_minima = 500 
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > area_minima]

    # Debemos tener al menos 4 objetos (las 4 referencias)
    if len(contornos_validos) < 4:
        print("Error: Se detectaron menos de 4 objetos. Asegúrate de que las referencias estén visibles.")
        return

    # 5. Obtener áreas para clasificación automática (K-Means)
    areas = np.array([cv2.contourArea(c) for c in contornos_validos], dtype=np.float32)
    
    # Necesitamos reformatear el array para cv2.kmeans
    datos_areas = areas.reshape(-1, 1)

    # Usamos K-Means para encontrar los 4 grupos de tamaño (clusters)
    # K=4 porque sabemos que hay monedas de 10, 5, 2 y 1
    k = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, etiquetas, centros = cv2.kmeans(datos_areas, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 'centros' contiene el área promedio de cada grupo.
    # Ordenamos los centros de menor a mayor área para saber cuál es cual.
    # flatten() convierte el array de [[area1], [area2]...] a [area1, area2...]
    centros_ordenados = sorted(centros.flatten())

    # Creamos un diccionario para mapear: Área Aproximada -> Valor Moneda
    # El centro más pequeño corresponde a $1, el siguiente a $2, etc.
    mapa_valores = {
        centros_ordenados[0]: 1,
        centros_ordenados[1]: 2,
        centros_ordenados[2]: 5,
        centros_ordenados[3]: 10
    }

    # Colores para identificar visualmente (B, G, R)
    colores = {
        1: (0, 0, 255),   # Rojo
        2: (255, 0, 0),   # Azul
        5: (0, 255, 255), # Amarillo
        10: (0, 255, 0)   # Verde
    }

    # 6. Clasificación y Conteo
    conteo = {1: 0, 2: 0, 5: 0, 10: 0}
    imagen_resultado = imagen_original.copy()

    # Iteramos sobre los contornos validos y sus etiquetas dadas por K-Means
    # zip une la lista de contornos con la lista de etiquetas que generó k-means
    for i, contorno in enumerate(contornos_validos):
        label_k_means = etiquetas[i][0] # Índice del grupo (0, 1, 2 o 3 sin orden específico)
        area_centro = centros[label_k_means][0] # El área promedio de ese grupo
        
        # Buscamos qué valor de moneda corresponde a esta área
        valor = mapa_valores[area_centro]
        
        conteo[valor] += 1
        
        # Dibujar contorno y texto
        cv2.drawContours(imagen_resultado, [contorno], -1, colores[valor], 3)
        
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(imagen_resultado, f"${valor}", (cX - 15, cY + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 7. Ajuste final: Restar las referencias
    # Como el usuario dijo que "siempre están" los círculos de control,
    # y son del mismo tamaño, el algoritmo los contó como monedas.
    # Restamos 1 a cada categoría.
    
    total_dinero = 0
    print("\n--- Resultado del Conteo (Excluyendo referencias) ---")
    for valor in [10, 5, 2, 1]:
        cantidad_real = max(0, conteo[valor] - 1) # Restamos la referencia, mínimo 0
        dinero = cantidad_real * valor
        total_dinero += dinero
        print(f"Monedas de ${valor}: {cantidad_real} detectadas (Total detectado {conteo[valor]} - 1 ref)")

    print("-----------------------------------------------------")
    print(f"DINERO TOTAL: ${total_dinero}")

    # Mostrar imagen
    cv2.imshow('Clasificacion por K-Means', imagen_resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Ejecución ---
if __name__ == "__main__":
    contar_monedas_con_referencia('test-img/18.png')