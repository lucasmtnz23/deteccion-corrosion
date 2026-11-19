
from pathlib import Path

from datasets import load_dataset

RUTA_DATOS = Path("data/corrosion-bi3q3")

if RUTA_DATOS.exists():
    print("El dataset ya está descargado.")
else:
    print("Descargando Corrosion-bi3q3...")

    dataset = load_dataset("Francesco/corrosion-bi3q3")
    dataset.save_to_disk(str(RUTA_DATOS))

    print("Dataset descargado correctamente.")
    for particion, datos in dataset.items():
        print(f"{particion}: {len(datos)} imágenes")
