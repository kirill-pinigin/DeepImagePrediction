import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from  ImagesRegressionCSVDataSet import  ImagesRegressionCSVDataSet , make_dataloaders

from DeepImagePredictor import DeepImagePredictor, IMAGE_SIZE, CHANNELS, DIMENSION
from ResidualPredictor import ResidualPredictor
from SqueezePredictors import  SqueezeSimplePredictor, SqueezeResidualPredictor, SqueezeShuntPredictor, SiLU

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type = str,   default='./koniq10k_224x224/', help='path to dataset')
parser.add_argument('--result_dir',     type = str,   default='./RESULTS/', help='path to result')
parser.add_argument('--predictor',      type = str,   default='ResidualPredictor', help='type of image generator')
parser.add_argument('--activation',     type = str,   default='SiLU', help='type of activation')
parser.add_argument('--criterion',      type = str,   default='MSE', help='type of criterion')
parser.add_argument('--optimizer',      type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--lr',             type = float, default='1e-3')
parser.add_argument('--split',          type = float, default='0.2')
parser.add_argument('--batch_size',     type = int,   default='16')
parser.add_argument('--epochs',         type = int,   default='101')
parser.add_argument('--augmentation',   type = bool,  default='False', help='type of training')
parser.add_argument('--pretrained',     type = bool,  default='True', help='type of training')

args = parser.parse_args()

predictor_types = { 'ResidualPredictor'                 : ResidualPredictor,
                    'SqueezeSimplePredictor'            : SqueezeSimplePredictor,
                    'SqueezeResidualPredictor'          : SqueezeResidualPredictor,
                    'SqueezeShuntPredictor'             : SqueezeShuntPredictor
                    }

activation_types = {'ReLU'     : nn.ReLU(),
                    'SSIMloss' : nn.LeakyReLU(),
                    'PReLU'    : nn.PReLU(),
                    'ELU'      : nn.ELU(),
                    'SELU'     : nn.SELU(),
                    'SiLU'     : SiLU()
              }

criterion_types = {
                    'MSE' : nn.MSELoss(),
                    'L1'  : nn.L1Loss(),
                    'BCE' : nn.BCEWithLogitsLoss()
                    }

optimizer_types = {
                    'Adam'           : optim.Adam,
                    'RMSprop'       : optim.RMSprop,
                    'SGD'           : optim.SGD
                    }

model = (predictor_types[args.predictor] if args.predictor in predictor_types else predictor_types['ResidualPredictor'])
function = (activation_types[args.activation] if args.activation in activation_types else activation_types['ReLU'])

predictor = model(dimension=DIMENSION , channels=CHANNELS, activation=function, pretrained = args.pretrained)

optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(predictor.parameters(), lr = args.lr)

criterion = (criterion_types[args.criterion] if args.criterion in criterion_types else criterion_types['MSE'])

train_transforms_list = [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=3),
        #transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.ToTensor(),
        ]

val_transforms_list = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=3),
        transforms.ToTensor(),
        ]

data_transforms = {
    'train':    transforms.Compose(train_transforms_list ),
    'val':      transforms.Compose(val_transforms_list),
}

image_dataset = ImagesRegressionCSVDataSet(os.path.join(args.data_dir, 'images'), csv_path = args.data_dir + 'scores.csv', channels = CHANNELS, transforms = data_transforms)

if args.split > float(0.0):
    dataloaders = make_dataloaders(image_dataset, batch_size = args.batch_size, splitratio = 0.2)
else:
    train_loader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, num_workers=4, shuffle= True)
    val_loader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, num_workers=4, shuffle= False)
    dataloaders = {'train' : train_loader , 'val' : val_loader}

framework = DeepImagePredictor(predictor = predictor, criterion = criterion, optimizer = optimizer,  directory = args.result_dir)
framework.fit(dataloaders = dataloaders, num_epochs=args.epochs)
framework.evaluate(dataloaders['val'])
