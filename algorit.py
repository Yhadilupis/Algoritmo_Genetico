import math
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import re
import shutil
import pandas as pd
from moviepy.editor import ImageSequenceClip


def calcular_x(a, i, delta_x):
    return a + i * delta_x


def calcular_fx(x):
    return x**2 + 3*x + 2


def graficar_generaciones(arreglo_dataframes, carpeta, a, b):
    # Generar puntos x dentro del intervalo [a, b] para la línea de la función
    x_line = np.linspace(a, b, 400)  # 400 puntos para una línea suave
    y_line = calcular_fx(x_line)

    for i, df in enumerate(arreglo_dataframes):
        # Asumiendo que cada DataFrame tiene columnas 'x' y 'f(x)' y están ordenados
        valores_x = df['x']
        valores_fx = df['f(x)']
        mejor_x = df.iloc[0]['x']
        mejor_fx = df.iloc[0]['f(x)']
        peor_x = df.iloc[-1]['x']
        peor_fx = df.iloc[-1]['f(x)']

        # Crear la gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(valores_x[1:-1], valores_fx[1:-1],
                   color='blue', label='Individuos')
        ax.scatter([mejor_x], [mejor_fx], color='green',
                   label='Mejor', zorder=5)
        ax.scatter([peor_x], [peor_fx], color='red', label='Peor', zorder=5)

        ax.plot(x_line, y_line, 'k--')

        ax.set_xlabel('Valor de x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Generación {i + 1}')
        ax.legend()
        ax.set_xlim(a, b)

        # Guardar la gráfica en la carpeta 'images'
        nombre_archivo = f'generacion_{i + 1}.png'
        ruta_archivo = os.path.join(carpeta, nombre_archivo)
        fig.savefig(ruta_archivo)
        plt.close(fig)


def main(p0, pmax, pmut, p_mut_gen, opt, num_gen, p, a, b):
    r = b - a

    num_pasos = math.ceil((r / p))
    num_saltos = num_pasos + 1

    bits = math.ceil(math.log2(num_saltos))
    delta_x = (r / (2**bits - 1))

    carpeta = 'imagenes'
    nombre_video = 'video_generaciones.mp4'

    if os.path.exists(carpeta):
        shutil.rmtree(carpeta)

    if os.path.exists(nombre_video):
        os.remove(nombre_video)

    os.makedirs(carpeta, exist_ok=True)

    molde = {
        'ID': [],
        'Individuo': [],
        'i': [],
        'x': [],
        'f(x)': [],
        'Generacion': []
    }

    estadisticas = pd.DataFrame(
        columns=['Generacion', 'Mejor', 'Peor', 'Promedio'])
    data = molde.copy()
    df = pd.DataFrame(data)

    # Generar población inicial
    id = 1
    generacion = 1
    for _ in range(p0):
        binario = ''.join(random.choice('01') for _ in range(bits))
        decimal = int(binario, 2)
        x = calcular_x(a, decimal, delta_x)
        nuevo_individuo = pd.DataFrame({
            'ID': [id],
            'Individuo': [binario],
            'i': [decimal],
            'x': [x],
            'f(x)': [calcular_fx(x)],
            'Generacion': [generacion]
        })
        df = pd.concat([df, nuevo_individuo], ignore_index=True)
        id += 1

    generaciones = []
    punto_cruza = bits // 2
    df = df.sort_values(by='f(x)', ascending=(opt == "MINIMIZAR"))

    for _ in range(num_gen):
        # Selección de los mejores
        mejor = df.iloc[0]['Individuo']
        resto = df.iloc[1:]['Individuo']

        # Cruzas
        nuevos = []
        for individuo in resto:
            padre1_parte1 = mejor[:punto_cruza]
            padre1_parte2 = mejor[punto_cruza:]
            padre2_parte1 = individuo[:punto_cruza]
            padre2_parte2 = individuo[punto_cruza:]
            hijo1 = padre1_parte1 + padre2_parte2
            hijo2 = padre2_parte1 + padre1_parte2
            nuevos.append(hijo1)
            nuevos.append(hijo2)

        # Mutación
        for index, hijo in enumerate(nuevos):
            numero_aleatorio = random.random()
            if numero_aleatorio <= pmut:
                hijo_lista = list(hijo)
                for i in range(len(hijo_lista)):
                    numero_aleatorio = random.random()
                    if numero_aleatorio <= p_mut_gen:
                        hijo_lista[i] = '1' if hijo_lista[i] == '0' else '0'
                nuevos[index] = ''.join(hijo_lista)

        for individuo in nuevos:
            decimal = int(individuo, 2)
            x = calcular_x(a, decimal, delta_x)
            nuevo_individuo = pd.DataFrame({
                'ID': [id],
                'Individuo': [individuo],
                'i': [decimal],
                'x': [x],
                'f(x)': [calcular_fx(x)],
                'Generacion': [generacion]
            })
            df = pd.concat([df, nuevo_individuo], ignore_index=True)
            id += 1

        df = df.sort_values(by='f(x)', ascending=(opt == "MINIMIZAR"))
        estadistico = pd.DataFrame({
            'Generacion': [generacion],
            'Mejor': [df.iloc[0]['f(x)']],
            'Peor': [df.iloc[-1]['f(x)']],
            'Promedio': [df['f(x)'].mean()]
        })

        # Guardando datos
        generaciones.append(df)
        estadisticas = pd.concat(
            [estadisticas, estadistico], ignore_index=True)
        # Eliminando duplicados
        df.drop_duplicates(subset=['Individuo'], keep='first', inplace=True)

        # Poda
        if len(df) > pmax:
            mejor = df.iloc[[0]]
            resto = df[1:].sample(n=pmax-1)
            df = pd.concat([mejor, resto], ignore_index=True)
        generacion += 1

    graficar_generaciones(generaciones, carpeta, a, b)
    crear_video(carpeta, nombre_video, 2)
    generaciones_grafica = estadisticas['Generacion']
    mejor = estadisticas['Mejor']
    peor = estadisticas['Peor']
    promedio = estadisticas['Promedio']

    # Crear la gráfica
    plt.figure(figsize=(12, 8))
    plt.plot(generaciones_grafica, mejor, label='Mejor', color='green')
    plt.plot(generaciones_grafica, peor, label='Peor', color='red')
    plt.plot(generaciones_grafica, promedio, label='Promedio', color='blue')

    # Añadir títulos y etiquetas
    plt.title('Mejor, Peor y Promedio por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Valor')
    plt.legend()

    # Mostrar la gráfica
    plt.show()


def crear_video(carpeta_imagenes, nombre_video, fps=2):
    rutas_imagenes = sorted(
        [os.path.join(carpeta_imagenes, img) for img in os.listdir(
            carpeta_imagenes) if img.endswith(".png")],
        key=ordenar_archivos_alphanumericamente
    )
    if not rutas_imagenes:
        print("No se encontraron imágenes en la carpeta especificada.")
        return
    clip = ImageSequenceClip(rutas_imagenes, fps=fps)
    clip.write_videofile(nombre_video)


def ordenar_archivos_alphanumericamente(archivo):
    numeros = re.findall(r'\d+', archivo)
    return int(numeros[0]) if numeros else 0


def ejecutar_algoritmo():
    try:
        # Obtener los valores ingresados por el usuario
        p0 = int(entries["p0"].get())
        pmax = int(entries["pmax"].get())
        pmut = float(entries["pmut"].get())
        p_mut_gen = float(entries["p_mut_gen"].get())
        num_gen = int(entries["num_gen"].get())
        p = float(entries["p"].get())
        a = float(entries["a"].get())
        b = float(entries["b"].get())
        opt = combo_opt.get()

        # Ejecutar el algoritmo
        main(p0, pmax, pmut, p_mut_gen, opt, num_gen, p, a, b)

    except ValueError:
        messagebox.showerror("Error", "Por favor, asegúrate de ingresar números válidos en todos los campos.")


# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Interfaz para Algoritmo Genético")

# Crear y colocar etiquetas y cuadros de entrada
etiquetas = ["p0", "pmax", "pmut", "p_mut_gen",
             "opt", "num_gen", "p", "a", "b"]
entries = {}

for i, etiqueta in enumerate(etiquetas):
    ttk.Label(ventana, text=etiqueta).grid(row=i, column=0, padx=5, pady=5)
    entrada = ttk.Entry(ventana)
    entrada.grid(row=i, column=1, padx=5, pady=5)
    entries[etiqueta] = entrada

# Configurar opciones para la variable 'opt'
opciones_opt = ["MINIMIZAR", "MAXIMIZAR"]
combo_opt = ttk.Combobox(ventana, values=opciones_opt)
combo_opt.grid(row=4, column=1, padx=5, pady=5)
combo_opt.set(opciones_opt[0])  # Establecer el valor predeterminado

# Botón para ejecutar el algoritmo
btn_ejecutar = ttk.Button(
    ventana, text="Ejecutar Algoritmo", command=ejecutar_algoritmo)
btn_ejecutar.grid(row=len(etiquetas), column=0, columnspan=2, pady=10)

# Iniciar el bucle de eventos
ventana.mainloop()
