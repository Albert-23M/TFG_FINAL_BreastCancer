import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Script for evaluating a RetinaNet model on COCO validation set.')

    parser.add_argument('--coco_path', help='Path to COCO directory', required=True)
    parser.add_argument('--model_path', help='Path to trained model', type=str, required=True)

    parser = parser.parse_args(args)

    print(f'Loading COCO dataset from: {parser.coco_path}')
    
    dataset_val = CocoDataset(parser.coco_path, set_name='coco_mass_test',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    print(f"Lengt of dataset: {len(dataset_val)}")

    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")


    # Cargar modelo
    if torch.cuda.is_available():
        retinanet = torch.load(parser.model_path)
    else:
        retinanet = torch.load(parser.model_path, map_location=torch.device('cpu'))

    if isinstance(retinanet, torch.nn.DataParallel):
        retinanet = retinanet.module  # Extrae modelo si estaba en DataParallel

    retinanet = torch.nn.DataParallel(retinanet).cuda() if torch.cuda.is_available() else torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()

    # Método alternativo para congelar BatchNorm si es necesario
    def freeze_bn(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

    retinanet.apply(freeze_bn)

    print("Evaluating model on COCO validation set...")
    coco_eval.evaluate_coco(dataset_val, retinanet)

if __name__ == '__main__':
    main()
