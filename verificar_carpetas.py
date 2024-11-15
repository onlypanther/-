import os

def contar_imagenes(ruta):
    total = 0
    for archivo in os.listdir(ruta):
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
            total += 1
    return total

# Rutas de las carpetas
ruta_bueno = 'datos_entrenamiento/bueno'
ruta_malo = 'datos_entrenamiento/malo'

print("\n=== Estado actual de las carpetas ===")
print("\nCarpeta 'bueno':")
print(f"- Contiene {contar_imagenes(ruta_bueno)} imágenes")
print("\nArchivos en 'bueno':")
for archivo in os.listdir(ruta_bueno):
    print(f"- {archivo}")

print("\nCarpeta 'malo':")
print(f"- Contiene {contar_imagenes(ruta_malo)} imágenes")
print("\nArchivos en 'malo':")
for archivo in os.listdir(ruta_malo):
    print(f"- {archivo}")

print("\n=== Resumen ===")
total_imagenes = contar_imagenes(ruta_bueno) + contar_imagenes(ruta_malo)
print(f"Total de imágenes para entrenamiento: {total_imagenes}")