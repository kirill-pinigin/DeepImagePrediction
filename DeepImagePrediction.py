import time
import sys
import os
import torch
from torch.autograd import Variable

LR_THRESHOLD = 1e-7
TRYING_LR = 5
DEGRADATION_TOLERANCY = 5
ACCURACY_TRESHOLD = float(0.0625)

class DeepImagePrediction(object):
    def __init__(self, predictor,  criterion, optimizer, dataloaders, directory, num_epochs=501):
        self.predictor = predictor
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_gpu = torch.cuda.is_available()
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders

        config = str(predictor.__class__.__name__) + '_' + str(predictor.activation.__class__.__name__)
        config += '_' + str(criterion.__class__.__name__)
        config += "_" + str(optimizer.__class__.__name__) #+ "_lr_" + str( optimizer.param_groups[0]['lr'])

        reportPath = os.path.join(directory, config + "/report/")
        flag = os.path.exists(reportPath)
        if flag != True:
            os.makedirs(reportPath)
            print('os.makedirs("reportPath")')

        self.modelPath = os.path.join(directory, config + "/model/")
        flag = os.path.exists(self.modelPath)
        if flag != True:
            os.makedirs(self.modelPath)
            print('os.makedirs("/modelPath/")')

        self.report = open(reportPath  + '/' + config + "_Report.txt", "w")
        _stdout = sys.stdout
        sys.stdout = self.report
        print(config)
        print(predictor)
        print(criterion)
        self.report.flush()
        sys.stdout = _stdout
        if self.use_gpu :
            self.predictor = self.predictor.cuda()
            #self.criterion = self.criterion.cuda()

    def __del__(self):
        self.report.close()

    def train(self):
        since = time.time()
        best_loss = 10000.0
        counter = 0
        i = int(0)
        degradation = 0
        for epoch in range(self.num_epochs):
            _stdout = sys.stdout
            sys.stdout = self.report
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            self.report.flush()
            sys.stdout = _stdout
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                self.dataloaders[phase].dataset.phase = phase
                if phase == 'train':
                    self.predictor.train(True)
                else:
                    self.predictor.train(False)

                running_loss = 0.0

                for data in self.dataloaders[phase]:
                    inputs, targets = data
                    if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        targets = Variable(targets.cuda())
                    else:
                        inputs, targets = Variable(inputs), Variable(targets)
                    self.optimizer.zero_grad()

                    outputs = self.predictor(inputs)
                    loss = self.criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.data[0] * inputs.size(0)

                epoch_loss = float(running_loss) / float(len(self.dataloaders[phase].dataset))

                _stdout = sys.stdout
                sys.stdout = self.report
                print('{} Loss: {:.4f} '.format(
                    phase, epoch_loss))
                self.report.flush()

                sys.stdout = _stdout
                print('{} Loss: {:.4f} '.format(
                    phase, epoch_loss))
                self.report.flush()

                if phase == 'val' and epoch_loss < best_loss:
                    counter = 0
                    degradation = 0
                    best_loss = epoch_loss
                    print('curent best_loss ', best_loss)
                    self.save('/BestPredictor.pth')
                else:
                    counter += 1
                    self.save('/RegualarPredictor.pth')

            if counter > TRYING_LR * 2:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 0.5
                        print('! Decrease LearningRate !', lr)
                counter = 0
                degradation += 1
            if degradation > DEGRADATION_TOLERANCY:
                print('This is the end! Best val best_loss: {:4f}'.format(best_loss))
                return best_loss

        time_elapsed = time.time() - since

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val best_loss: {:4f}'.format(best_loss))
        return best_loss

    def evaluate(self, test_loader, modelPath = None):
        if modelPath is not None:
            self.predictor.load_state_dict(torch.load(modelPath))
            print('load predictor model')
        else:
            self.predictor.load_state_dict(torch.load(self.modelPath + 'BestPredictor.pth'))
            print('load BestSRgenerator ')

        since = time.time()
        self.predictor.train(False)
        self.predictor.eval()
        if self.use_gpu:
            self.predictor = self.predictor.cuda()
        running_loss = 0.0
        for data in test_loader:
            inputs, targets = data

            if self.use_gpu:
                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())
            else:
                inputs, targets = Variable(inputs), Variable(targets)

            outputs = self.generator(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss.data[0] * inputs.size(0)

            _stdout = sys.stdout
            sys.stdout = self.report
            print(' loss ', float(loss[0]))
            self.report.flush()

            sys.stdout = _stdout
            print(' loss ', float(loss[0]))
            self.report.flush()

        epoch_loss = float(running_loss) / float(len(test_loader.dataset))

        time_elapsed = time.time() - since

        print('Evaluating complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('epoch_loss: {:4f}'.format(epoch_loss))

    def save(self, model):
        self.predictor = self.predictor.cpu()
        self.predictor.eval()
        torch.save(self.predictor.state_dict(), self.modelPath + '/' + model)
        if self.use_gpu:
            self.predictor = self.predictor.cuda()
        self.predictor.train()
