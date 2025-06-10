## Comandos e instrucciones para el entrenamiento

Primero, debemos situarnos en la raíz del proyecto y activar el entorno virtual.

-------------------------------------

Una vez activado, para comenzar el entrenamiento se debe ejecutar el siguiente comando:


*python train.py --dataset coco --coco_path /retinanet/pytorch-retinanet/coco --depth numDepth*

El valor numDepth debe ser uno de los siguientes: 18, 34, 50, 101, 152.

También se pueden especificar otros parámetros, como por ejemplo el número de épocas (--epochs), el tamaño del batch (--batch_size), entre otros.
## Comandos e instrucciones para validación

-------------------------------------
Al igual que para el entrenamiento, primero hay que activar el entorno virtual.
Para validar, situados en la raíz del proyecto, ejecutamos:

*python coco_validation.py --coco_path /home/albert/research/retinanet/pytorch-retinanet/coco --model_path /pytorch-retinanet/nombreModeloFinal.pt*

## Información directorio COCO
El directorio COCO está vacío, ya que ahí es donde se guardan las imágenes para train, validation y testing. Lo mismo pasa con los COCO, están vacíos porque es el COCO de cada directorio de imágenes.

## Configuración entorno
Para configurar el entorno, hay que utilizar el archivo .yml, que define todas las dependencias necesarias.
*conda env create -f environment.yml*

## Link de los modelos entrenados
https://drive.google.com/file/d/1z4jaKLjMl_-qXIokAtnCYtejcSwKTVte/view?usp=sharing
