import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime


ruta = 'empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

for nombre in lista_empleados:
    img_actual = cv2.imread(f'{ruta}\{nombre}')
    mis_imagenes.append(img_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])
    
print(nombres_empleados)

def codificar(img):
    lista_codificada = []
    
    for i in img:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        codificaciones = fr.face_encodings(i)
        if len(codificaciones) > 0:
            codificado = codificaciones[0]
            lista_codificada.append(codificado)
        else:
            print("No se encontró ninguna cara en una de las imágenes.")
        
    return lista_codificada

# registrar ingresos
def registro(persona):
    f = open('registro.csv', 'r+')
    lista_datos = f.readlines()
    nombre_registro = []
    
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombre_registro.append(ingreso[0])
    
    if persona not in nombre_registro:
        hora = datetime.now()
        str_hora = hora.strftime('%H:%M:%S')
        f.writelines(f'\n{persona}, {str_hora}')

lista_empleados_codificada = codificar(mis_imagenes)
print(len(lista_empleados_codificada))

# tomar una imagen desde la cámara del portatil
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# leer la img de la cámara
exito, imagen = captura.read()

if not exito:
    print('No se ha reconocido ninguna cara')
else:
    captura = fr.face_locations(imagen)
    
    captura_codificada = fr.face_encodings(imagen, captura)
    
    # buscar coincidencias
    for caraCODE, caraUBI in zip(captura_codificada, captura):
        coincidencias = fr.compare_faces(lista_empleados_codificada, caraCODE)
        distancia = fr.face_distance(lista_empleados_codificada, caraCODE)
        
        print(distancia)
        
        indice_coincidencia = numpy.argmin(distancia)
        
        # mostrar coincidencias si las hay
        if distancia[indice_coincidencia] > .6:
            print("No coincide con ningun empleado")
        else:
            # buscar el nombre del empleado
            nombre = nombres_empleados[indice_coincidencia]
            
            y1, x2, y2, x1 = caraUBI
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imagen, nombre, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            registro(nombre)     
                   
            # mostrar la imagen obtenida
            cv2.imshow('Imagen web', imagen)
            
            # mantener ventana abierta
            cv2.waitKey(0)