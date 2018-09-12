import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from  ImagesRegressionCSVDataSet import  ImagesRegressionCSVDataSet , make_validation_split

from DeepImagePrediction import DeepImagePrediction
from SqueezePredictors import  SqueezePredictor

class SiLU(torch.nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        out = torch.mul(x, F.sigmoid(x))
        return out


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type = str,   default='./koniq10k_224x224/', help='path to dataset')
parser.add_argument('--result_dir',     type = str,   default='./RESULTS/', help='path to result')
parser.add_argument('--predictor',      type = str,   default='StanfordImageTransformer', help='type of image generator')
parser.add_argument('--activation',     type = str,   default='ReLU', help='type of activation')
parser.add_argument('--criterion',      type = str,   default='MSE', help='type of criterion')
parser.add_argument('--optimizer',      type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--lr',             type = float, default='1e-3')
parser.add_argument('--dimension',       type = int,   default='1')
parser.add_argument('--channels',       type = int,   default='3')
parser.add_argument('--image_size',     type = int,   default='224')
parser.add_argument('--batch_size',     type = int,   default='4')
parser.add_argument('--epochs',         type = int,   default='256')
parser.add_argument('--augmentation',   type = bool,  default='True', help='type of training')
parser.add_argument('--pretrained',     type = bool,  default='True', help='type of training')

args = parser.parse_args()


predictor_types = { 'SqueezePredictors' : SqueezePredictor  }


activation_types = {'ReLU'     : nn.ReLU(),
                    'SSIMloss' : nn.LeakyReLU(),
                    'PReLU'    : nn.PReLU(),
                    'ELU'      : nn.ELU(),
                    'SELU'     : nn.SELU(),
                    'SiLU'     : SiLU
              }

criterion_types = {
                    'MSE': nn.MSELoss(),
                    'L1' : nn.L1Loss()
                    }

optimizer_types = {
                    'Adam'           : optim.Adam,
                    'RMSprop'       : optim.RMSprop,
                    'SGD'           : optim.SGD
                    }

model = (predictor_types[args.predictor] if args.predictor in predictor_types else predictor_types['SqueezePredictors'])
function = (activation_types[args.activation] if args.activation in activation_types else activation_types['ReLU'])

predictor = model(dimension=args.dimension , channels=args.channels, activation=function)

optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(predictor.parameters(), lr = args.lr)

criterion = (criterion_types[args.criterion] if args.criterion in criterion_types else criterion_types['MSE'])

minimal_transforms_list =[
        transforms.Resize((args.image_size, args.image_size), interpolation=3),
        transforms.ToTensor(),
    ]

testing_transforms_list = [
    transforms.Resize((args.image_size, args.image_size), interpolation=3),
    transforms.ToTensor(),
]

if args.channels == 1:
    minimal_transforms_list = [transforms.Grayscale()] + minimal_transforms_list
    testing_transforms_list = [transforms.Grayscale()] + testing_transforms_list

data_transforms = {
    'train':    transforms.Compose(minimal_transforms_list if args.augmentation == False
                                    else [ transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] + minimal_transforms_list),
    'val':      transforms.Compose(minimal_transforms_list),
    'test':     transforms.Compose(testing_transforms_list)
}

image_datasets = ImagesRegressionCSVDataSet(os.path.join(args.data_dir, 'images'), csv_path = args.data_dir + 'scores.csv', channels = args.channels, transforms = data_transforms)

dataloaders = make_validation_split(image_datasets, batch_size = args.batch_size, ratio = 0.2)

framework = DeepImagePrediction(predictor = predictor, criterion = criterion, optimizer = optimizer, dataloaders = dataloaders, num_epochs=args.epochs, directory = args.result_dir)
framework.train()
framework.evaluate(dataloaders['test'])
