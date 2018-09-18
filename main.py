import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from  ImagesRegressionCSVDataSet import  ImagesRegressionCSVDataSet , make_dataloaders

from DeepImagePrediction import DeepImagePrediction
from SqueezePredictors import  SqueezePredictor, SqueezeResidualPredictor, SqueezeShuntPredictor, SiLU

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type = str,   default='./koniq10k_224x224/', help='path to dataset')
parser.add_argument('--result_dir',     type = str,   default='./RESULTS/', help='path to result')
parser.add_argument('--predictor',      type = str,   default='SqueezeShuntPredictor', help='type of image generator')
parser.add_argument('--activation',     type = str,   default='SiLU', help='type of activation')
parser.add_argument('--criterion',      type = str,   default='MSE', help='type of criterion')
parser.add_argument('--optimizer',      type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--lr',             type = float, default='1e-3')
parser.add_argument('--split',          type = float, default='0.2')
parser.add_argument('--dimension',       type = int,   default='1')
parser.add_argument('--channels',       type = int,   default='3')
parser.add_argument('--image_size',     type = int,   default='224')
parser.add_argument('--batch_size',     type = int,   default='16')
parser.add_argument('--epochs',         type = int,   default='101')
parser.add_argument('--augmentation',   type = bool,  default='False', help='type of training')
parser.add_argument('--pretrained',     type = bool,  default='True', help='type of training')

args = parser.parse_args()

predictor_types = { 'SqueezePredictor'         : SqueezePredictor,
                    'SqueezeResidualPredictor' : SqueezeResidualPredictor,
                    'SqueezeShuntPredictor'    : SqueezeShuntPredictor
                    }

activation_types = {'ReLU'     : nn.ReLU(),
                    'SSIMloss' : nn.LeakyReLU(),
                    'PReLU'    : nn.PReLU(),
                    'ELU'      : nn.ELU(),
                    'SELU'     : nn.SELU(),
                    'SiLU'     : SiLU()
              }

criterion_types = {
                    'MSE': nn.MSELoss(),
                    'L1' : nn.L1Loss(),
                    }

optimizer_types = {
                    'Adam'           : optim.Adam,
                    'RMSprop'       : optim.RMSprop,
                    'SGD'           : optim.SGD
                    }

model = (predictor_types[args.predictor] if args.predictor in predictor_types else predictor_types['SqueezePredictor'])
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

image_dataset = ImagesRegressionCSVDataSet(os.path.join(args.data_dir, 'images'), csv_path = args.data_dir + 'scores.csv', channels = args.channels, transforms = data_transforms)

if args.split > float(0.0):
    dataloaders = make_dataloaders(image_dataset, batch_size = args.batch_size, splitratio = 0.2)
else:
    train_loader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, num_workers=4, shuffle= True)
    val_loader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, num_workers=4, shuffle= False)
    dataloaders = {'train' : train_loader , 'val' : val_loader}

framework = DeepImagePrediction(predictor = predictor, criterion = criterion, optimizer = optimizer, dataloaders = dataloaders, num_epochs=args.epochs, directory = args.result_dir)
framework.train()
framework.evaluate(dataloaders['val'])
