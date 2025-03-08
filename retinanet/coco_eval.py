import torch
import json
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pycocotools.cocoeval import COCOeval

def draw_boxes(image, boxes, labels, scores=None, color=(0, 255, 0)):
    for i, box in enumerate(boxes):
        x, y, w, h = map(int, box)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = str(labels[i])
        if scores is not None:
            label += f' {scores[i]:.2f}'
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def calculate_iou(box1, box2):
    """
    Calcula el IoU (Intersection over Union) entre dos bounding boxes.
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

def evaluate_coco_validation(dataset, model, iou_threshold=0.75):
    pass


def evaluate_coco(dataset, model, iou_threshold=0.75):
    print("---------------------------------------------------------------------------------------------------------------------------")
    model.eval()
    with torch.no_grad():
        thresholds = [round(x, 2) for x in np.arange(0.01, 1.00, 0.1)]
        results = {threshold: {'tp': 0, 'fp': 0, 'num_total_gt_bboxes': 0} for threshold in thresholds}
        
        num_images = len(dataset)
        random_indices = [3, 4]  # random.sample(range(len(dataset)), 2)
        wantToSave = True
        saved_images = 0
        exp_name = "exp_1"
        print("---------------------------------------------------------------------------------------------------------------------------")
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
            
            if boxes.shape[0] > 0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
            
            gt_boxes = data['annot'][:, :4].cpu().numpy()
            for idx in range(gt_boxes.shape[0]):
                gt_boxes[idx][2] = gt_boxes[idx][2] - gt_boxes[idx][0]
                gt_boxes[idx][3] = gt_boxes[idx][3] - gt_boxes[idx][1]
            
            gt_labels = data['annot'][:, 4].cpu().numpy()
            num_total_gt_bboxes = len(gt_boxes)
            print("---------------------------------------------------------------------------------------------------------------------------")
            for threshold in thresholds:
                tp = 0
                fp = 0
                
                valid_indices = scores >= threshold
                filtered_boxes = boxes[valid_indices]
                filtered_labels = labels[valid_indices]
                filtered_scores = scores[valid_indices]
                
                iou_values = calculate_iou(gt_boxes, filtered_boxes * scale)
                
                for i, ious in enumerate(iou_values):
                    max_iou = max(ious) if len(ious) > 0 else 0
                    if max_iou >= iou_threshold:
                        tp += 1
                    else:
                        fp += 1
                print("---------------------------------------------------------------------------------------------------------------------------")
                results[threshold]['tp'] += tp
                results[threshold]['fp'] += fp
                results[threshold]['num_total_gt_bboxes'] += num_total_gt_bboxes
            
            img = data['img'].numpy()
            img = (img - img.min())*(255/(img.max() - img.min()))
            img = img.astype(np.uint8)
            
            gt_image = draw_boxes(img.copy(), gt_boxes, gt_labels, color=(255, 0, 0))
            pred_image = draw_boxes(img.copy(), boxes * scale, labels, scores, color=(0, 255, 0))
            print("---------------------------------------------------------------------------------------------------------------------------")
            if wantToSave and (index in random_indices) and (saved_images < 2):
                cv2.imwrite(f'/home/albert/research/retinanet/pytorch-retinanet/resultados_imagenes/{exp_name}/ground_truth_{index}.jpg', gt_image)
                cv2.imwrite(f'/home/albert/research/retinanet/pytorch-retinanet/resultados_imagenes/{exp_name}/prediction_{index}.jpg', pred_image)
                iou_values = calculate_iou(gt_boxes, (boxes * scale))
                saved_images += 1
            
            print(f'Procesando imagen {index+1}/{num_images}', end='\r')
    
    model.train()
    return results

