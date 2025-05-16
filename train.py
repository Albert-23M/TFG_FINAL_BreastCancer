import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import AugmenterFlipX, AugmenterCutMix, AugmenterFlipY, CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Normalizer
from torch.utils.data import DataLoader
from retinanet import coco_eval, csv_eval
import matplotlib.pyplot as plt

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Crear dataset y dataloader
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        
        dataset_train = CocoDataset(parser.coco_path, set_name='optimam_coco_mass_train',
                                    transform=transforms.Compose([Normalizer(), AugmenterFlipX(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='optimam_coco_mass_val',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None or parser.csv_classes is None:
            raise ValueError('Must provide --csv_train and --csv_classes when training on CSV dataset.')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), AugmenterFlipX(), Resizer()]))
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()])) if parser.csv_val else None
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=8, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=8, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Crear el modelo
    model_dict = {18: model.resnet18, 34: model.resnet34, 50: model.resnet50, 101: model.resnet101, 152: model.resnet152}
    if parser.depth not in model_dict:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    # Para usar Transfer Learning, sustituimos esta linea por la que hay debajo de esta
    # retinanet = model_dict[parser.depth](num_classes=dataset_train.num_classes(), pretrained=True)
    # print(type(retinanet)) # es <class 'retinanet.model.ResNet'>

    retinanet_pretrained = torch.load('/home/albert/research/retinanet/pytorch-retinanet_old_version/model_final_full_mammo_REAL_cbis-ddsm.pt')
    retinanet = retinanet_pretrained.module # Retiramos el módulo DataParallel
    
    # print(type(retinanet_pretrained)) # es <class 'torch.nn.parallel.data_parallel.DataParallel'>


    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    epoch_losses = []

    
    thresholds = np.logspace(-3, 0, 50)
    # Diccionarios para guardar el historial
    training_history = {"train_loss": [], "val_loss": [], "avg_precision": []}
    val_data_history = {threshold: {'tp': [], 'fp': [], 'fn': [], 'tpr': [], 'fppi': [], 'num_total_gt_bboxes': []} for threshold in thresholds}



    best_ap = 0.0  # Mejor AP encontrada
    epochs_no_improve = 0  # Contador de épocas sin mejora
    patience = 20  # Número de épocas permitidas sin mejora

    print(f"Lengt of dataset_train: {len(dataset_train)}")
    print(f"Lengt of dataset_val: {len(dataset_val)}")

    # batch = next(iter(dataloader_train))

    # # Plot all images in the batch
    # imgs = batch["img"]
    # batch_size = imgs.shape[0]
    # fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns

    # for i in range(batch_size):
    #     img = imgs[i].permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    #     img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    #     row, col = divmod(i, 4)
    #     axes[row, col].imshow(img)
    #     axes[row, col].axis('off')
    #     axes[row, col].set_title(f"Image {i+1}")

    # plt.tight_layout()
    # plt.savefig('batch_images_optimam.png')
    # # plt.show()
    # return


    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                classification_loss, regression_loss = classification_loss.mean(), regression_loss.mean()
                loss = classification_loss + regression_loss

                if loss == 0:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                epoch_loss.append(float(loss))

                del classification_loss, regression_loss
            except Exception as e:
                print(e)
                continue

        mean_train_loss = np.mean(epoch_loss)
        training_history["train_loss"].append(mean_train_loss)
        print(f'Epoch {epoch_num} | Train Loss: {mean_train_loss:.5f}')

        # Evaluación en el dataset de validación
        if dataset_val:
            print('Evaluating dataset...')
            if parser.dataset == 'coco':
                val_metrics = coco_eval.evaluate_coco(dataset_val, retinanet)
                
                if isinstance(val_metrics, dict) and "AP" in val_metrics:
                    avg_precision = val_metrics["AP"]  # Usamos la métrica principal de AP
                else:
                    avg_precision = compute_avg_precision(val_metrics)
            
            elif parser.dataset == 'csv':
                val_metrics = csv_eval.evaluate(dataset_val, retinanet)
                avg_precision = compute_avg_precision(val_metrics)

            for threshold in thresholds:
                val_data_history[threshold]['tp'].append(val_metrics[threshold]['tp'])
                val_data_history[threshold]['fp'].append(val_metrics[threshold]['fp'])
                val_data_history[threshold]['fn'].append(val_metrics[threshold]['fn'])
                val_data_history[threshold]['tpr'].append(val_metrics[threshold]['tpr'])
                val_data_history[threshold]['fppi'].append(val_metrics[threshold]['fppi'])
                val_data_history[threshold]['num_total_gt_bboxes'].append(val_metrics[threshold]['num_total_gt_bboxes'])

            training_history["avg_precision"].append(avg_precision)
            print(f'Epoch {epoch_num} | Validation AP: {avg_precision:.5f}')


            # *** Early Stopping Logic ***
            if avg_precision > best_ap:
                best_ap = avg_precision
                epochs_no_improve = 0  # Reset counter
                torch.save(retinanet.module, 'best_model_ddsm_TL_to_optimam_ES.pt')  # Guardar mejor modelo
                print(f'🔹 New best model saved with AP: {best_ap:.5f}')
            else:
                epochs_no_improve += 1
                print(f'No improvement for {epochs_no_improve} epochs')

            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch_num}')
                break  # Sale del loop de entrenamiento
            
        scheduler.step(mean_train_loss)
        torch.save(retinanet.module, f'{parser.dataset}_retinanet_{epoch_num}.pt')

    retinanet.eval()
    torch.save(retinanet, 'model_final_cbis-ddsm_TL_to_optimam_ES.pt')
    # print(val_data_history)

    # Graficar la pérdida de entrenamiento y validación
    plot_training_history(training_history)
    plot_avg_precision(training_history)


def plot_avg_precision(training_history):
    plt.figure(figsize=(10, 5))
    
    epochs = range(len(training_history["avg_precision"]))
    
    plt.plot(epochs, training_history["avg_precision"], label='Average Precision', marker='o', color='green')
    
    plt.xlabel('Epochs')
    plt.ylabel('Average Precision')
    plt.title('Average Precision Over Epochs')
    plt.legend()
    plt.grid()
    
    plt.savefig('avg_precision_history_ddsm_TL_to_optimam_ES.png')
    # plt.show()
    plt.close()

def plot_training_history(training_history):
    plt.figure(figsize=(10, 5))
    
    epochs = range(len(training_history["train_loss"]))
    
    plt.plot(epochs, training_history["train_loss"], label='Train Loss', marker='o')
    if len(training_history["val_loss"]) > 0:
        plt.plot(epochs, training_history["val_loss"], label='Validation Loss', marker='o')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    
    plt.savefig('training_history_ddsm_TL_to_optimam_ES.png')
    # plt.show()
    plt.close()

def compute_avg_precision(val_loss):
    if not val_loss:
        return np.nan  # Si no hay datos, evitar errores
    
    precisions = []
    
    for threshold, metrics in val_loss.items():  # Iteramos sobre los valores del diccionario
        tp = metrics.get("tp", 0)  # Si no hay "tp", asigna 0
        fp = metrics.get("fp", 0)  # Si no hay "fp", asigna 0

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else np.nan  # Evitar división por 0

if __name__ == '__main__':
    main()

