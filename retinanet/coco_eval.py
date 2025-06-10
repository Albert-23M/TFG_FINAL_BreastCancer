import csv
import torch
import json
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pycocotools.cocoeval import COCOeval
import pickle
from sklearn.metrics import auc

def draw_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_labels, scores, ious):
    """
    Dibuja las ground truths y predicciones en la misma imagen, incluyendo los valores de IoU.
    Ahora los textos se dibujan en una esquina de la imagen como una leyenda.
    
    - `gt_boxes`: Bounding boxes del ground truth (azul).
    - `gt_labels`: Labels de ground truth.
    - `pred_boxes`: Bounding boxes predichas por el modelo (verde).
    - `pred_labels`: Labels de predicción.
    - `scores`: Confianza de las predicciones.
    - `ious`: Lista con los valores de IoU para cada predicción.
    """
    
    legend = []  # Lista para almacenar las leyendas

    # Dibujamos ground truth en azul
    for i, box in enumerate(gt_boxes):
        x, y, w, h = map(int, box)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        legend.append(f'GT {i+1}')  # Agregar a la leyenda

    # Convertimod ious (matriz) a una lista plana del max IoU por predicción
    ious_per_pred = [max(col) if len(col) > 0 else 0.0 for col in zip(*ious)]

    for i, box in enumerate(pred_boxes):
        x, y, w, h = map(int, box)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde
        iou_value = ious_per_pred[i] if i < len(ious_per_pred) else 0.0
        legend.append(f'Pred {i+1}: IoU={iou_value:.2f}')


    # Dibujar la leyenda en la esquina superior izquierda
    x_offset, y_offset = 10, 20  # Posición inicial de la leyenda
    for i, text in enumerate(legend):
        y_position = y_offset + i * 20
        cv2.putText(image, text, (x_offset, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image

def calculate_iou(box1, box2):
    """
    Calculate the IoU (Intersection over Union) between two lists of bounding boxes.
    Each box is in the format (x, y, w, h).
    Returns a matrix of shape (len(box1), len(box2)) with the IoU for each pair.
    """
    iou_matrix = [[0 for _ in range(len(box2))] for _ in range(len(box1))]
    for i, b1 in enumerate(box1):
        for j, b2 in enumerate(box2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            x1_min, y1_min = x1, y1
            x1_max, y1_max = x1 + w1, y1 + h1
            x2_min, y2_min = x2, y2
            x2_max, y2_max = x2 + w2, y2 + h2
            x_inter_min = max(x1_min, x2_min)
            y_inter_min = max(y1_min, y2_min)
            x_inter_max = min(x1_max, x2_max)
            y_inter_max = min(y1_max, y2_max)
            inter_width = max(0, x_inter_max - x_inter_min)
            inter_height = max(0, y_inter_max - y_inter_min)
            inter_area = inter_width * inter_height
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area != 0 else 0
            iou_matrix[i][j] = iou
    return iou_matrix

def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on a set of boxes.
    
    Args:
        boxes (np.array): Array of bounding boxes in the format (x, y, w, h).
        scores (np.array): Array of confidence scores for each box.
        iou_threshold (float): Threshold for IoU. Boxes with IoU >= this value are suppressed.
        
    Returns:
        keep (list): List of indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []
    
    # Convert to float if necessary
    boxes = boxes.astype(np.float32)
    
    # Sort the boxes by score (highest first)
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        if indices.size == 1:
            break
        
        # Compute IoU between the current box and all the remaining boxes
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        xx1 = np.maximum(current_box[0], other_boxes[:, 0])
        yy1 = np.maximum(current_box[1], other_boxes[:, 1])
        xx2 = np.minimum(current_box[0] + current_box[2], other_boxes[:, 0] + other_boxes[:, 2])
        yy2 = np.minimum(current_box[1] + current_box[3], other_boxes[:, 1] + other_boxes[:, 3])
        
        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h
        
        area_current = current_box[2] * current_box[3]
        areas = other_boxes[:, 2] * other_boxes[:, 3]
        union_area = area_current + areas - inter_area
        ious = inter_area / union_area
        
        # Keep boxes with IoU less than the threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep

def evaluate_coco(dataset, model, iou_threshold=0.1, nms_iou_threshold=0.5):
    """
    Evaluate the COCO validation dataset with the given model.
    Now integrated with non maximum suppression (NMS).
    
    Args:
        dataset: The dataset to evaluate.
        model: The detection model.
        iou_threshold (float): IoU threshold for matching predictions to ground truths.
        nms_iou_threshold (float): IoU threshold used in the NMS step.
    
    Returns:
        results (dict): A dictionary with the results for each score threshold.
    """
    zero_pred_count = 0
    num_images = len(dataset)
    model.eval()
    with torch.no_grad():
        thresholds = np.linspace(0,1,500)
        results = {threshold: {'tp': 0, 'fp': 0, 'fn': 0, 'tpr': 0, 'fppi': 0, 'num_total_gt_bboxes': 0} for threshold in thresholds}
        
        # Para guardar imágenes de ejemplo
        random_indices = [0,1,2,3,4,5] # random.sample(range(len(dataset)), 6)
        wantToSave = False

        saved_images = 0
        
        all_scores = []  # Lista para almacenar todos los scores

        all_fppi_values = []
        all_tpr_values = []

        correct_gt_boxes = set()
        incorrect_gt_boxes = set()
        
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
                
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()
            boxes /= scale 
            
            all_scores.extend(scores) 
            if scores.shape[0] == 0:
                zero_pred_count += 1
            
            if boxes.shape[0] > 0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
            
            # Prepare ground truth boxes and labels
            gt_boxes = data['annot'][:, :4].cpu().numpy()
            for idx in range(gt_boxes.shape[0]):
                gt_boxes[idx][2] = gt_boxes[idx][2] - gt_boxes[idx][0]
                gt_boxes[idx][3] = gt_boxes[idx][3] - gt_boxes[idx][1]
            gt_labels = data['annot'][:, 4].cpu().numpy()
            num_total_gt_bboxes = len(gt_boxes)
            
            best_iou_sum = -1
            best_filtered_boxes = None
            best_iou_values = None
            best_labels = None
            best_scores = None

            for threshold in thresholds:
                tp = 0
                fp = 0
                fn = 0
                
                clipped_scores = np.clip(scores, 0.0, 1.0)

                # Se filtran las predicciones por el umbral actual
                valid_indices = clipped_scores > threshold
                filtered_boxes = boxes[valid_indices]
                filtered_labels = labels[valid_indices]
                filtered_scores = clipped_scores[valid_indices]
                
                # Se aplica non_maximum suppression (NMS) a las predicciones filtradas
                keep_indices = non_maximum_suppression(filtered_boxes, filtered_scores, iou_threshold=nms_iou_threshold)
                
                filtered_boxes = filtered_boxes[keep_indices]
                filtered_labels = filtered_labels[keep_indices]
                filtered_scores = filtered_scores[keep_indices]
                
                # Multiplicamos por scale para que las boxes estén en la escala original
                iou_values = calculate_iou(gt_boxes, filtered_boxes * scale)
                matched_gt_indices = set()
                matched_preds = set()

                # Para cada ground_truth box, miramos si hay alguna predicción que tenga un IoU >= iou_threshold
                for i, ious in enumerate(iou_values):
                    max_iou, max_iou_ind = 0, -1
                    for j, iou in enumerate(ious):
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_ind = j
                    if max_iou >= iou_threshold:
                        tp += 1
                        matched_gt_indices.add(i)
                        matched_preds.add(max_iou_ind)
                        

                fp = len(iou_values[0]) - len(matched_preds)
                fn = num_total_gt_bboxes - len(matched_gt_indices)

                results[threshold]['tp'] += tp
                results[threshold]['fp'] += fp
                results[threshold]['fn'] += fn
                results[threshold]['num_total_gt_bboxes'] += num_total_gt_bboxes

                iou_sum = sum(max(iou_values[i]) for i in matched_gt_indices)
                if iou_sum > best_iou_sum:
                    best_iou_sum = iou_sum
                    best_filtered_boxes = filtered_boxes
                    best_iou_values = iou_values
                    best_labels = filtered_labels
                    best_scores = filtered_scores

            # Calculamos el iou de los gt y pred finales:
            if best_filtered_boxes is not None and len(best_filtered_boxes) > 0:
                iou_values = calculate_iou(gt_boxes, best_filtered_boxes * scale)
            else:
                iou_values = []

            # Quiero saber si para cada gt_boxes existe al menos un best_filtered_boxes con iou >= iou_threshold
            matched_gt_indices = set()
            for i, ious in enumerate(iou_values):
                for j, iou in enumerate(ious):
                    if iou >= iou_threshold:
                        matched_gt_indices.add(i)
                        break
            
            # Actualizar los sets de correctos e incorrectos
            for i in range(len(gt_boxes)):
                if i in matched_gt_indices:
                    correct_gt_boxes.add(tuple(gt_boxes[i]))
                else:
                    incorrect_gt_boxes.add(tuple(gt_boxes[i]))

            if wantToSave and (index in random_indices) and (saved_images < 6):
                # Codigo opcional para guardar imágenes de ejemplo
                img = data['img'].numpy()
                img = (img - img.min()) * (255 / (img.max() - img.min()))
                img = img.astype(np.uint8)
                
                # Generar imagen con GT y predicciones juntas
                comparison_image = draw_boxes(img.copy(), gt_boxes, gt_labels, best_filtered_boxes * scale, best_labels, best_scores, best_iou_values)

                # Guardar imágenes solo si están en los índices aleatorios y quedan menos de 6 guardadas
                cv2.imwrite(f'image_comparison_{index}_cbis-ddsm_SISI_TL_To_Optimam.jpg', comparison_image)
                saved_images += 1
            
            print(f'Processing image {index+1}/{num_images}', end='\r')
    
    # Calcular tpr y FPPI al final para cada threshold
    for threshold in thresholds:
        tp = results[threshold]['tp']
        fp = results[threshold]['fp']
        fn = results[threshold]['fn']
        num_total_gt_bboxes = results[threshold]['num_total_gt_bboxes']
        
        results[threshold]['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[threshold]['fppi'] = fp / num_images if num_images > 0 else 0

        all_fppi_values.append(results[threshold]['fppi'])
        all_tpr_values.append(results[threshold]['tpr'])
  

    fppi = np.array(all_fppi_values)
    tpr = np.array(all_tpr_values)

    # Ordenamos por FPPI
    sorted_indices = np.argsort(fppi)
    fppi = fppi[sorted_indices]
    tpr = tpr[sorted_indices]

    # Guardamos los resultados en un archivo CSV para posteriormente graficar
    save_bboxes_to_csv(correct_gt_boxes, 'gt_pred_correct.csv', scale)
    save_bboxes_to_csv(incorrect_gt_boxes, 'gt_pred_incorrect.csv', scale)

    model.train()
    
    return results

def save_bboxes_to_csv(bboxes, csv_filename, scale=1.0):
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x1', 'y1', 'x2', 'y2', 'width', 'height', 'scale'])

        for x1, y1, w, h in bboxes:
            x2 = x1 + w
            y2 = y1 + h
            writer.writerow([x1, y1, x2, y2, w, h, scale])


def plot_tpr_fppi(results):
    thresholds = sorted(results.keys())
    tpr_values = [results[t]['tpr'] for t in thresholds]
    fppi_values = [results[t]['fppi'] for t in thresholds]

    plt.figure(figsize=(8, 6))
    plt.plot(fppi_values, tpr_values, linestyle='-', color='b', label="TPR@FPPI")

    plt.xlabel("False Positives Per Image (FPPI)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("TPR vs FPPI Curve")
    # plt.ylim(0.5, 1.0)
    #plt.xlim(0.0, 0.1)
    plt.grid(True)
    plt.legend()
    plt.savefig('tpr_vs_fppi_optimam_to_cbis-ddsm_plot_fix_clipped_scores.png')
    plt.close()



def plot_score_histogram(all_scores):
    """
    Plotea y guarda un histograma de los scores de todas las predicciones.

    :param all_scores: Lista con todos los scores de las predicciones.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(all_scores, bins=50, color='blue', alpha=0.7, edgecolor='black')

    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Scores")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("score_histogram_CBIS-DDSM_SISI_TL_to_OPTIMAM_ES_threshold_0.1.png")
    # plt.show()
    plt.close()
