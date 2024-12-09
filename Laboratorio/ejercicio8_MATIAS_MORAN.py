import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import math
import imageio as img
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def descompone_imagen(image):
    R = image[:,:,0] # Matriz de rojos
    G = image[:,:,1] # Matriz de verdes
    B = image[:,:,2] # Matriz de azules
    return(R,G,B)


image = img.imread(os.getcwd() + '\\arbol.jpg', format='jpg')
R, G, B = descompone_imagen(image)

plt.imshow(G,cmap='gray',vmin=0,vmax=255)
plt.savefig('imagen_original.png')

U, s, V = npl.svd(G)
S = np.zeros((G.shape[0], G.shape[1]))
S[:len(s), :len(s)] = np.diag(s)
G_ = np.dot(U,np.dot(S,V))

plt.clf()
plt.imshow(G_, cmap='gray', vmin=0, vmax=255)
plt.savefig('imagen_recompuesta.png')

#si comparamos las 2 imagenes vemos que son identicas


def reduce_svd(A,p):
    U,s,V = npl.svd(A)
    n_elementos = int(p*len(s))
    s[(len(s) - n_elementos):len(s)] = 0
    S = np.zeros((A.shape[0], A.shape[1]))
    S[:len(s), :len(s)] = np.diag(s)
    A_ = np.dot(U,np.dot(S,V))
    return(A_)


G_reducida = reduce_svd(G, 0.9)
plt.clf()
plt.imshow(G_reducida, cmap='gray', vmin=0, vmax=255)
plt.savefig('imagen_gris_reducida.png')



def comprime_svd(imagen,p):
    R,G,B = descompone_imagen(imagen)
    R_ = reduce_svd(R,p)
    G_ = reduce_svd(G,p)
    B_ = reduce_svd(B,p)
    imagen_comprimida = np.zeros((R_.shape[0],R_.shape[1],3))
    imagen_comprimida[:,:,0] = R_
    imagen_comprimida[:,:,1] = G_
    imagen_comprimida[:,:,2] = B_
    return(imagen_comprimida)




def errores_de_compresion(image):
    error = []
    p_values = np.arange(0, 1, .02)
    for p in p_values:
        print('Calculando con p='+str(p))
        image_ = comprime_svd(image,p)
        error.append(np.mean(np.abs(image_-image))/np.mean(image))
    return np.array(error)


print(errores_de_compresion(image))
for p in np.arange(0,1, .1):
    plt.clf()
    plt.imshow(comprime_svd(image,p).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.savefig('imagen_color_reducida_'+str(p)+'.png')

# por lo observado en las imagenes reduciendo de a un 10%, si queremos conservar la imagen casi igual a la original podemos usar solamente un 20% de
# los valores principales


plt.clf()
plt.figure()
p_values = np.arange(0, 1, .02)
imagenes = ['arbol.jpg', 'mona_lisa.jpg', 'fractal.jpg', 'poligono.jpeg', 'cuadrado.jpg']
for imagePath in imagenes:
    print(f'Trabajando en imagen {imagePath}')
    image = img.imread(os.getcwd() + '\\' + imagePath, format='jpg')
    errores = errores_de_compresion(image)
    plt.plot(p_values, errores, label=imagePath)
plt.legend()
plt.savefig('errores.png')

# como podemos observar en el grafico, mientras mas compleja es la imagen, peor es su calidad de compresion al reducir los PV que quitamos
# en casos como el cuadrado la compresion casi no afecta a la calidad y es tolerante a una gran compresion