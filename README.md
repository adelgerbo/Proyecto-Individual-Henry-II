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



