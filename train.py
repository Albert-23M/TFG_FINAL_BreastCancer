import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
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
        
        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None or parser.csv_classes is None:
            raise ValueError('Must provide --csv_train and --csv_classes when training on CSV dataset.')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()])) if parser.csv_val else None
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=4, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=4, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Crear el modelo
    model_dict = {18: model.resnet18, 34: model.resnet34, 50: model.resnet50, 101: model.resnet101, 152: model.resnet152}
    if parser.depth not in model_dict:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    retinanet = model_dict[parser.depth](num_classes=dataset_train.num_classes(), pretrained=True)

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    epoch_losses = []

    # Diccionario para guardar el historial
    training_history = {"train_loss": [], "val_loss": []}

    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                # plot con las bbox
                break
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                classification_loss, regression_loss = classification_loss.mean(), regression_loss.mean()
                loss = classification_loss + regression_loss

                if loss == 0:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                # print(f'Epoch: {epoch_num} | Iter: {iter_num} | Cls loss: {classification_loss:.5f} | '
                  #     f'Reg loss: {regression_loss:.5f} | Loss: {np.mean(loss_hist):.5f}')

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
                val_loss = coco_eval.evaluate_coco(dataset_val, retinanet)  # Devuelve alguna métrica?
            elif parser.dataset == 'csv':
                val_loss = csv_eval.evaluate(dataset_val, retinanet)

            training_history["val_loss"].append(val_loss if val_loss is not None else 0)

        scheduler.step(mean_train_loss)
        torch.save(retinanet.module, f'{parser.dataset}_retinanet_{epoch_num}.pt')

    retinanet.eval()
    torch.save(retinanet, 'model_final.pt')

    # Graficar la pérdida de entrenamiento y validación
    plot_training_history(training_history)

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
    
    plt.savefig('training_history.png')
    plt.show()

if __name__ == '__main__':
    main()
