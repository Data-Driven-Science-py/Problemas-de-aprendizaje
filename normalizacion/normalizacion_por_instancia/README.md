# Normalizacion por instancia

![image1](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYG1kTDa93D9CvoBMzruaZcv8WGCyIClO95A&s)

Aunque no directamente implementado con este objectivo, es innegable que la [normalizacion por baches](https://arxiv.org/pdf/1502.03167) tiene un efecto regulador en el proceso de entrenamiento de modelos avanzados. Fue primeramente creado con el objetivo de estabilizar las distribuciones de las salidas de capas secuenciales en redes neuronales. A medida que la entrada se propaga hacia delante, su distribucion se desvia lo suficiente como para generar una discrepancia entre las normas de gradientes de capas secuenciales que imposibilita un entrenamiento uniforme a traves de toda la estructura. Es por esto que se creo la normalizacion por baches, para poder solucionar los cambios de distribucion repentinos a medida que se propaga la entrada.

## Formulacion matematica

La formulacion matematica se encuentra en el pdf: formulacionMatematica.pdf

