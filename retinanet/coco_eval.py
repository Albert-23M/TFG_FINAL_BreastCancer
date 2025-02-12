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

def evaluate_coco(dataset, model, threshold=0.05):
    model.eval()
    with torch.no_grad():
        results = []
        image_ids = []
        random_indices = random.sample(range(len(dataset)), 10)
        saved_images = 0
        
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
                
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()
            # boxes /= scale

            if boxes.shape[0] > 0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if score < threshold:
                        break

                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }
                    results.append(image_result)

            image_ids.append(dataset.image_ids[index])
            print('{}/{}'.format(index, len(dataset)), end='\r')

            # Guardar imágenes con bboxes (10 imágenes aleatorias)
            
            if index in random_indices and saved_images < 2:
                img = data['img'].numpy()
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                gt_boxes = data['annot'][:, :4].cpu().numpy()
                gt_labels = data['annot'][:, 4].cpu().numpy()

                pred_boxes = boxes.numpy()
                pred_labels = labels.numpy()
                pred_scores = scores.numpy()

                print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                print("Image shape ==> ", img.shape)
                print("gt_boxes =======>>>> ", gt_boxes)
                print()
                print("pred_boxes =============>  ", pred_boxes)
                '''
                gt_image = draw_boxes(img.copy(), gt_boxes, gt_labels, color=(255, 0, 0))
                pred_image = draw_boxes(img.copy(), pred_boxes, pred_labels, pred_scores, color=(0, 255, 0))

                cv2.imwrite(f'ground_truth_{index}.jpg', gt_image)
                cv2.imwrite(f'prediction_{index}.jpg', pred_image)
                '''
                saved_images += 1
            

        if not results:
            return

        json.dump(results, open(f'{dataset.set_name}_bbox_results.json', 'w'), indent=4)
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(f'{dataset.set_name}_bbox_results.json')
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    model.train()
    return
