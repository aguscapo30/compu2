from PIL import Image
import multiprocessing as mp
import numpy as np
import signal
import time
import sys
from scipy.ndimage import gaussian_filter 

def cargar_imagen(ruta_imagen):
    imagen = Image.open(ruta_imagen)
    return imagen

def dividir_imagen(imagen, n):
    imagen_np = np.array(imagen)
    partes = np.array_split(imagen_np, n, axis=0)
    return partes

def aplicar_filtro(parte_imagen):
    return gaussian_filter(parte_imagen, sigma=2)

def procesar_parte(parte, shared_memory, shape, index, conn):
    try:
        resultado = aplicar_filtro(parte)
        shared_array = np.frombuffer(shared_memory.get_obj()).reshape(shape)
        shared_array[index:index+parte.shape[0], :, :] = resultado
        conn.send('done')
        conn.close()
    except (KeyboardInterrupt, SystemExit):
        print("Proceso Worker interrumpido. Terminando de manera controlada.")
        sys.exit(0)

def proceso_coordinador(pipes, event, n):
    try:
        for i in range(n):
            pipes[i][0].recv()
            pipes[i][0].close()
        event.set()
    except (KeyboardInterrupt, SystemExit):
        print("Proceso coordinador interrumpido. Terminando de manera controlada.")
        event.set()
        sys.exit(0)

def crear_procesos_y_procesar(shared_memory, shape, partes, event):
    n = len(partes)
    pipes = [mp.Pipe() for _ in range(n)]
    procesos = [
        mp.Process(target=procesar_parte, args=(partes[i], shared_memory, shape, sum(part.shape[0] for part in partes[:i]), pipes[i][1]))
        for i in range(n)
    ]
    proceso_coord = mp.Process(target=proceso_coordinador, args=(pipes, event, n))
    
    for proceso in procesos:
        proceso.start()

    proceso_coord.start()

    try:
        for proceso in procesos:
            proceso.join()
        proceso_coord.join()
    except (KeyboardInterrupt, SystemExit):
        print("Interrupción recibida. Terminando subprocesos y proceso coordinador de manera controlada.")
        event.set()
        for proceso in procesos:
            proceso.terminate()
        proceso_coord.terminate()
        sys.exit(0)

def guardar_imagen(shared_memory, shape, ruta_salida):
    imagen_completa = np.frombuffer(shared_memory.get_obj()).reshape(shape)
    imagen_final = Image.fromarray(np.uint8(imagen_completa))
    imagen_final.save(ruta_salida)

def proceso_principal(shared_memory, shape, start_time, image_output, event):
    try:
        event.wait()
        guardar_imagen(shared_memory, shape, image_output)
        total_time = time.time() - start_time
        print(f'Tiempo total: {total_time}')
    except (KeyboardInterrupt, SystemExit):
        print("Interrupción en el proceso principal. Terminando de manera controlada.")
        sys.exit(0)

def manejador_senal(sig, frame):
    print(f'Señal {sig} recibida. Terminando el programa de manera controlada.')
    raise SystemExit(0)

if __name__ == "__main__":
    # MANEJANDO LAS SEÑALES
    signal.signal(signal.SIGINT, manejador_senal)

    filtro_seleccionado = 'gaussian'
    num_partes = 8

    start_time = time.time()

    image_name = '/Users/francosoldatilopez/Desktop/manu/umcomp2/tp1/imagen.jpeg'  
    image_output = '/Users/francosoldatilopez/Desktop/manu/umcomp2/tp1/imagen_filtrado.jpeg'  
    imagen = cargar_imagen(image_name)
    partes = dividir_imagen(imagen, num_partes)
    
    # MEMORIA
    altura_total = sum(part.shape[0] for part in partes)
    ancho = partes[0].shape[1]
    profundidad = partes[0].shape[2]

    shape = (altura_total, ancho, profundidad)

    shared_memory = mp.Array('d', int(np.prod(shape)))
    event = mp.Event()

    # PROCESOS
    principal = mp.Process(target=proceso_principal, args=(shared_memory, shape, start_time, image_output, event))
    secundario = mp.Process(target=crear_procesos_y_procesar, args=(shared_memory, shape, partes, event))

    principal.start()
    secundario.start()

    try:
        secundario.join()
        principal.join()
    except (KeyboardInterrupt, SystemExit):
        print("Interrupción en el proceso principal. Terminando todos los procesos de manera controlada.")
        event.set()
        principal.terminate()
        secundario.terminate()
        sys.exit(0)