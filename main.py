import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn

from  ImagePredictionCSVDataSet import  ImagePredictionCSVDataSet

from DeepImagePrediction import DeepImagePrediction
from MobilePredictor import MobilePredictor
from MnasPredictor import MnasPredictor
from ResidualPredictor import ResidualPredictor
from SqueezePredictors import  SqueezeSimplePredictor, SqueezeResidualPredictor, SqueezeShuntPredictor
from NeuralModels import SILU

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir',      type = str,   default='./AdjustPhotoDataset256/', help='path to dataset')
parser.add_argument('--result_dir',     type = str,   default='./RESULTS/', help='path to result')
parser.add_argument('--predictor',      type = str,   default='MobilePredictor', help='type of image generator')
parser.add_argument('--activation',     type = str,   default='ReLU', help='type of activation')
parser.add_argument('--criterion',      type = str,   default='BCE', help='type of criterion')
parser.add_argument('--optimizer',      type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--lr',             type = float, default=1e-3)
parser.add_argument('--l2',             type = float, default=0)
parser.add_argument('--batch_size',     type = int,   default=80)
parser.add_argument('--epochs',         type = int,   default=64)
parser.add_argument('--resume_train',   type = bool,  default=True, help='type of training')

args = parser.parse_args()

predictor_types = { 'ResidualPredictor'        : ResidualPredictor,
                    'MobilePredictor'          : MobilePredictor,
                    'MnasPredictor'            : MnasPredictor,
                    'SqueezeSimplePredictor'   : SqueezeSimplePredictor,
                    'SqueezeResidualPredictor' : SqueezeResidualPredictor,
                    'SqueezeShuntPredictor'    : SqueezeShuntPredictor
                    }

activation_types = {'ReLU'      : nn.ReLU(),
                    'LeakyReLU' : nn.LeakyReLU(),
                    'PReLU'     : nn.PReLU(),
                    'ELU'       : nn.ELU(),
                    'SELU'      : nn.SELU(),
                    'SILU'      : SILU()
              }

criterion_types = {
                    'MSE' : nn.MSELoss(),
                    'L1'  : nn.L1Loss(),
                    'BCE' : nn.BCELoss(),
                    }

optimizer_types = {
                    'Adam'     : optim.Adam,
                    'RMSprop'  : optim.RMSprop,
                    'SGD'      : optim.SGD
                    }

model = (predictor_types[args.predictor] if args.predictor in predictor_types else predictor_types['ResidualPredictor'])
function = (activation_types[args.activation] if args.activation in activation_types else activation_types['ReLU'])
predictor = model(activation=function)
optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(predictor.parameters(), lr = args.lr, weight_decay=args.l2)
criterion = (criterion_types[args.criterion] if args.criterion in criterion_types else criterion_types['MSE'])

augmentations = {'train' : True, 'val' : False}
shufles = {'train' : True, 'val' : False}

if  'koniq10k_224x224' in args.image_dir:
    print(args.image_dir)
    image_datasets = {x: ImagePredictionCSVDataSet(os.path.join(args.image_dir, 'images'), csv_path=args.image_dir +'/'+ x + '.csv', augmentation=augmentations[x]) for x in ['train', 'val']}

else:
    image_datasets = {x: ImagePredictionCSVDataSet(os.path.join(args.image_dir, x), csv_path=args.image_dir + '/' + x + '/' + x + '.csv', augmentation=augmentations[x])for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=shufles[x], num_workers=4)
                for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(dataloaders['val'], batch_size=1, shuffle=False, num_workers=4)

framework = DeepImagePrediction(predictor = predictor, criterion = criterion, optimizer = optimizer,  directory = args.result_dir)
framework.approximate(dataloaders = dataloaders, num_epochs=args.epochs, resume_train=args.resume_train)
framework.estimate(testloader)
