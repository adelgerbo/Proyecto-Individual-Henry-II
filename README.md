<h1 align=center> HENRY’S LABS </h1>

<h2 align=center>PROYECTO INDIVIDUAL II -- DATA SCIENCE<br>
    Alejandro del Gerbo Actis</h2>


## **Temática**

El proyecto propuesto puede verse en su totalidad en el [siguiente link](https://github.com/adelgerbo/Proyecto-Individual-Henry-II/blob/main/CONSIGNAS.md).

Como resumen, se nos entregó un dataset en formato parquet con 346.000 registros correspondientes a avisos de propiedades en alquier en Estados Unidos. A los efectos del proyecto, tomariamos el valor del alquiler como si se tratara del valor de venta de las propiedades, separariamos los registros en dos clases, propiedades de valor "alto" y propiedades de valor "bajo".

Debiamos comenzar haciendo un Analisis Exploratorio de los Datos [(EDA)](https://www.ibm.com/ar-es/cloud/learn/exploratory-data-analysis), realizando luego las transformaciones que consideraramos necesarias, para generar luego un modelo de [Machine Learning Supervisado](https://universidadeuropea.com/blog/aprendizaje-supervisado-no-supervisado/#:~:text=El%20modelo%20que%20se%20utiliza,de%20los%20conjuntos%20de%20datos) que pueda predecir con un alto grado de precisión al recibir el dataset provisto de testeo con 38.000 registros (y que no contenía la información del precio) si las propiedades allí contenidas eran del segmento de valor "bajo".

Por otro lado, se requería desarrollar un modelo de Machine Learning No Supervisado, que dividiera el dataset de testeo en 3 categorías.

En ambos modelos, se generarían archivos en formato csv con las predicciones, los cuales luego se enviarían a Henry para que se los valorara en cuanto a la Exactitud [(Accuracy)](https://developers.google.com/machine-learning/crash-course/classification/accuracy?hl=es-419) del Modelo Supervisado y por la métrica de [Silhouette](https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c) para el caso del Modelo No Supervisado.



## **Tareas Realizadas**

### Librerias utilizadas
* [Pandas](https://pandas.pydata.org/) y [Numpy](https://numpy.org/) para la exploración, transformación y manipulación de los datos
* [Seaborn](https://seaborn.pydata.org/) y [Matplotlib](https://matplotlib.org/) para las visualizaciones
* [Category Encoders](https://contrib.scikit-learn.org/category_encoders/) para el encoding de variables categóricas
* [Sklearn](https://scikit-learn.org/stable/) para utilizar sus modelos de Machine Learnin, escalado de datos, validación cruzada, división del set en entrenamiento y testeo y métricas para evaluar los modelos

### EDA (Analisis exploratorio de datos)
Luego de cumplir con la primera consigna que consistía en insertar una nueva columna con el número 1 en las propiedaes cuyo valor era inferior o igual a 999 dólares y con el número 0 para el resto, y eliminar la columna "price", revisamos la distribución de ambas clases en el conjunto de datos, y comprobamos que se encontraban balanceadas, por lo que no requerían tratamiento en ese aspecto.

Notamos luego que había en el dataset una gran cantidad de anuncios similares, en los cuales podía llegar a haber alguna variación de precio (que podría deberse a la inflación reinante en Estados Unidos), pero en las que generalmente el resto de los datos, e incluso el texto con la descripción eran exactamente iguales. Confirmó nuestras sospechas el hecho de que la URL que direccionaba hacia la imagen de la propiedad, era siempre la misma.

Utilizando la columna que contenía dicha URL, detectamos 178.233 anuncios repetidos que correspondian a las mismas propiedades, practicamente la mitad del dataset. Considerando que esta repetición afectaría negativamente nuestros modelos, por el hecho de que ciertas características tendrían mas peso en el análisis solo por el hecho de haberse publicado mas veces esas propiedades, decidimos eliminar esa columna, con una salvedad: Pensamos que podría servirnos el dato referente a cuantas veces apareció publicada cada propiedad (tal vez las propiedades de mayor valor tardan mas tiempo en alquilarse), por lo que generamos una nueva columna conteniendo esa información llamada "publicaciones".

Notamos luego la existencia de ouliers en varias columnas.
* "sqfeet"(pies cuadrados de la propiedad): Encontramos valores en cero y muy bajos, y valores que nos parecieron extraordinariamente altos. Debido a nuestra ignorancia en el mercado de Real Estate de EEUU, buscamos información online, dentro de lo cual accedimos a esta [página](https://www.ahs.com/home-matters/real-estate/the-2022-american-home-size-index/), gracias a la cual obtuvimos el dato de que el promedio de pies cuadrados de las viviendas en Estados Unidos se acerca a 2.000.
Por lo tanto, consideramos los valores superiores a 50.000 como errores (especialmente tratandose de alquileres).
Tambien consideraremos errores los valores de cero y muy bajos, ya que de acuerdo a este otro [artículo](https://www.nyrentownsell.com/blog/square-footage-guide-to-living-real-life-home-example/), 200 pies cuadrados es el mínimo requerido para que la vivienda sea habitable.
Utilizamos el método del Rango Intercuartílico para fijar los lìmites desde los cuales considerariamos los outliers e hicimos algunas gráficas, pero no consideramos adecuados esos valores. Tomamos como valor mínimo 200 pies cuadrados y como máximo 10.000, ajustando los outliers a los mismos.
* "publicaciones": En esta columna creada por nosotros encontramos propiedades con mas de 100 avisos repetidos en el dataset, por lo que determinamos que como máximo tomariamos la cantidad de 30. Ajustamos y graficamos.
* "beds" y "baths": Graficamos ambas con y sin outliers, y decidimos ajustar los mismos a los valores mínimos y máximos de 1 y 8.
* "long"(lonmgitud, coordenada): Encontramos valores en positivo que no correspondian a Estados Unidos, pero verificamos con Google Maps que invirtiendo el signo, dichos puntos se ubicaban en California. Transformamos entonces los valores positivos en negativos. Luego verificamos cuales eran los límites de longitud del territorio del país, y ajustamos los registros que se excedian a esos valores. Por último, detectamos 918 valores nulos, los cuales decidimos reemplazar por el valor promedio.

### Variables Categóricas
Los modelos de Machine Learning que usaríamos aceptan únicamente valores numéricos dentro de las features con las cuales se los entrena, por lo que debíamos analizar las variables categóricas exitentes para efectuar transformaciones con ellas:
* "Laundry Options" (Servicio de Lavanderia): En primer lugar, detectamos 33.294 valores nulos, y presumiendo que de no existir un dato allí correpondería a que no contaban con el servicio, los reemplazamos por "No laundry on site". Luego asignamos valores del 0 al 4 para las distintos conceptos, ordenando de menor a mayor y comenzando desde las que no tienen lavandería en el lugar hasta las que contaban con lavadora y secadora dentro del inmueble.
* "Parking Options" (Estacionamiento): En primer lugar, detectamos 55.694 valores nulos, y presumiendo que de no existir un dato allí correpondería a que no contaban con estacionamiento, los reemplazamos por "no parking". Luego asignamos valores del 0 al 4 para las distintos conceptos, ordenando de menor a mayor, comenzando desde las que no tienen estacionamiento o se estaciona en la calle, y terminando con las que contaban con valet parking, agrupando aquellos conceptos de valuación similar (carport y detached garage por ejemplo).
* Analizamos tambien el tipo de vivienda en búsqueda de correlación, graficamos y detectamos que algunos tipos de propiedades tenian mayor proporción de precios altos que otros. Pasamos entonces las variables categóricas asignandoles numeros del 0 al 5, agrupando en este paso tambien algunas categorías de similar valoración (duplex y loft por ejemplo).
* Con respecto al Estado, vimos en esta [publicación](https://www.fool.com/the-ascent/research/average-house-price-state/#:~:text=The%20median%20home%20price%20in,in%20the%20U.S.%20at%20%24354%2C649.) que el Estado en el cual se ubican las propiedades tiene gran influencia sobre el precio promedio de las mismas. Por lo tanto, decidimos utilizar la columna "state" en el modelos, pero necesitabamos transformarla en valores numéricos. Hicimos esto mediante el método de [Binary Encoding](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/), ya que el [One-Hot Encoding](https://www.analyticsvidhya.com/blog/2021/05/how-to-perform-one-hot-encoding-for-multi-categorical-variables/) nos crearía demasiadas nuevas columnas.

### Análisis de correlaciones
Comenzamos analizando si las coordenadas de los inmuebles podrían tener alguna correlación con el precio, y vimos que en la latitud la correlaciñon era practicamente nula, pero que en el caso de la longitud, existia cierta correlación. Atribuimos esto a que en Estados Unidos la mayor parte de la población reside sobre las costas del Pacífico y del Atlántico, por lo que las propiedades cercanas a las mismas, deberían tener mayor demanda y por ende, mayor. Pero por otro lado, no existe tal diferencia en cuanto al norte o sur del país. Por lo tanto, decidimos eliminar la columan de Latitud.

Antes de continuar con el analisis de correlaciones, decidimos eliminar algunas columnas mas:
* El ID y la URL del anuncio no aportan ningún dato de interes
* Existian en el dataset mas de 400 regiones, pero al contar con el Estado, desechamos el dato de Región y Region URL
* La URL con la imagen del inmueble es un dato que se podría utilizar mediante software de reconociento de imagenes, pero consideramos que no contabamos con el tiempo suficiente para poder cumplir con la deadline, por lo que la eliminamos.
* Con el mismo criterio eliminamos la descripción del inmueble, que requería la implementación de un modelo de [procesamiento de lenguaje natural](https://www.aprendemachinelearning.com/procesamiento-del-lenguaje-natural-nlp/).

Procedimos luego a graficar las correlaciones de todas las features que aun permanecian en el dataframe, con la clase objetivo. De acuerdo al gráfico, determinamos que comenzariamos utlizando Bedrooms, Baths, Parking Options, Laundry Options, Square Feet, Publicaciones, Longitud y Permitido Fumar.









