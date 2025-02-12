import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)
    print(" ************************************** ", parser.coco_path)
    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        saved_model = torch.load(parser.model_path)
        if isinstance(saved_model, torch.nn.DataParallel):
            saved_model = saved_model.module  # Extrae el modelo del contenedor DataParallel
        retinanet.load_state_dict(saved_model.state_dict())
        # retinanet.load_state_dict(torch.load(parser.model_path, weights_only=True))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()
    # retinanet.freeze_bn()
    coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()
