# GENERACION DE NOMBRES DE DINOSAURIOS A PARTIR DE UNA RED NEURONAL RECURRENTE

import numpy as np
np.random.seed(5)

from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K

# 1. Lectura del set de datos
nombres = open('nombres_dinosaurios.txt','r').read()
nombres = nombres.lower()

# Creamos un diccionario para almacenar el alfabeto del que se componen todas las palabras del dataset
alfabeto = list(set(nombres))
tam_datos, tam_alfabeto = len(nombres), len(alfabeto)
print("En total hay %d caracteres, y el diccionario tiene un tamano de %d caracteres." % (tam_datos, tam_alfabeto))

# Conversion de caracteres a indices y viceversa
car_a_ind = {car:ind for ind, car in enumerate(sorted(alfabeto))}
ind_a_car = {ind:car for ind, car in enumerate(sorted(alfabeto))}
#print(car_a_ind)
#print(ind_a_car)

# 2. MODELO
n_a = 25    # Numero de neuronas en la capa oculta
entrada = Input(shape=(None, tam_alfabeto))
a0 = Input(shape=(n_a,))

celda_recurrente = SimpleRNN(n_a, activation='tanh', return_state=True)
capa_salida = Dense(tam_alfabeto, activation='softmax')

salida = []
hs, _ = celda_recurrente(entrada, initial_state=a0)
salida.append(capa_salida(hs))
modelo = Model([entrada, a0], salida)
modelo.summary()

opt = SGD(lr=0.0005)
modelo.compile(optimizer=opt, loss='categorical_crossentropy')

# 3. EJEMPLOS DE ENTRENAMIENTO

# Crear lista con ejemplos de entrenamiento y mezclarla aleatoriamente
with open("nombres_dinosaurios.txt") as f:
    ejemplos = f.readlines()
ejemplos = [x.lower().strip() for x in ejemplos]
np.random.shuffle(ejemplos)

# Crear ejemplos de entrenamiento usando un generador
def train_generator():
    while True:
        # Tomar un ejemplo aleatorio
        ejemplo = ejemplos[np.random.randint(0,len(ejemplos))]

        # Convertir el ejemplo a representacion numerica
        X = [None] + [car_a_ind['\n']]

        # Crear "Y", resultado de desplazar "X" un caracter a la derecha
        Y = X[1:] + [car_a_ind['\n']]

        # Representar "X" y "Y" en formato one-hot
        x = np.zeros((len(X),1,tam_alfabeto))
        onehot = to_categorical(X[1:],tam_alfabeto).reshape(len(X)-1,1,tam_alfabeto)
        x[1:,:,:] = onehot
        y = to_categorical(Y, tam_alfabeto).reshape(len(X),tam_alfabeto)

        # Activacion inicial (matriz de ceros)
        a = np.zeros((len(X), n_a))

        yield [x,a],y

# 4. ENTRENAMIENTO
BATCH_SIZE = 80     # de ejemplos de entrenamiento a usar en cada iteracion
NITS = 10000        # # de iteraciones

for j in range(NITS):
    historia = modelo.fit_generator(train_generator(), steps_per_epoch=BATCH_SIZE, epochs=1, verbose=0)

    # Imprimir evolucion del entrenamiento cada 1000 iteraciones
    if j%1000 == 0:
        print('\nIteracion: %d, Error: %f' % (j,historia.history['loss'][0]) + '\n')


# 5. GENERACION DE NOMBRES USANDO EL MODELO ENTRENADO
def generar_nombre(modelo, car_a_num, tam_alfabeto, n_a):
    # Inicializar x y a con ceros
    x = np.zeros((1,1,tam_alfabeto,))
    a = np.zeros((1,n_a))

    # Nombre generado y caracter de fin de linea
    nombre_generado = ''
    fin_linea = '\n'
    car = -1

    # Iterar sobre el modelo y generar prediccion hasta tanto no se alcance
    # "fin_linea" o el nombre generado llegue a los 50 caracteres
    contador = 0
    while (car != fin_linea and contador != 50):
        # Generar prediccion usando la celda RNN
        a, _ = celda_recurrente(K.constant(x), initial_state=K.constant(a))
        y = capa_salida(a)
        prediccion = K.eval(y)
        
        # Escoger aleatoriamente un elemento de la prediccion
        # (El elemento con probabilidad mas alta tendra mas opciones de ser seleccionado)
        ix = np.random.choice(list(range(tam_alfabeto)), p = prediccion.ravel())
        
        # Convertir el elemento seleccionado a caracter y anadirlo al nombre generado
        car = ind_a_car[ix]
        nombre_generado += car
        
        # Crear x_(t+1) = y_t y a_t = a_(t-1)
        x = to_categorical(ix, tam_alfabeto).reshape(1,1,tam_alfabeto)
        a = K.eval(a)
        
        # Actualizar contador y continuar
        contador += 1
        
        # Agregar fin de linea al nombre generado en caso de tener mas de 50 caracteres
        if (contador == 50):
            nombre_generado += '\n'
    
    print("Nombre generado por la RNN: ", nombre_generado)
    
# Generar 100 ejemplos de nombres generados por el modelo ya entrenado
for i in range(100):
    generar_nombre(modelo, car_a_ind, tam_alfabeto, n_a)