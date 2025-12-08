import torch
import torch.nn as nn
import numpy as np
# from assemble import collateList
from classes import SepsisModel
from classes import SepsisDataset
from torch.utils.data import DataLoader
import sys
import sklearn.metrics as metrics

def confusionMatrix(x, y, yPred): #to be used in validation mode
    # print(x[0], x[0].shape)
    # print(y[0], y[0].shape)
    # print(yPred[0], yPred[0].shape)
    cutoff=0.35
    yPredClass = np.zeros_like(yPred[0])
    yPredClass[yPred[0] > cutoff] = 1 #add ones where cutoff breached
    # print(yPredClass)
    print(metrics.confusion_matrix(y[0], yPredClass))

#default params
maxEpochs = 10
mode = 'train' #need to make trianing and test mode
#use torch.save and torch.load
learningRate = 0.0001

#custom params
arguments = sys.argv
for argument in arguments:
    argList = argument.split('=')
    if (argList[0] == 'epochs'):
        maxEpochs = int(argList[1])
    if (argList[0] == 'mode'):
        mode = argList[1]
    if (argList[0] == 'lr'):
        learningRate = float(argList[1])
print("Running neural network in", mode, "mode with", maxEpochs, "epochs and lr =", learningRate)

#Collates UIDs for training, validation, test sets
trainingUIDs = np.loadtxt('trainingSet.psv', delimiter='|', dtype=np.str)
print("trainingUIDs: " + str(trainingUIDs[:-1]))
validationUIDs = np.loadtxt('validationSet.psv', delimiter='|', dtype=np.str)
print("validationUIDs: " + str(validationUIDs[:-1]))

# testUID = '106498'

#need to try and use as many cols as possible
# colSequence = (0, 2, 5, 6, 3, 39)
# colSequence = (0, 2, 5, 6, 3, 1, 11, 12, 15, 23, 28, 21, 39)
colSequence = (0, 1, 2, 4, 5, 6, 10, 19, 26, 33, 39)
colSequence = (0, 1, 2, 4, 5, 6, 39)
# colSequence = (0, 1, 2, 3, 4, 5, 6, 10, 11, 15, 21, 23, 28, 39)
# colSequence = (0, 1, 2, 3, 5, 6, 11, 12, 15, 21, 23, 28, 39, 10, 33, 26, 4, 19)
#using many high nan% cols results in most tensors having one row
#one row tensors are likely useless - discard?
#need to make sure values used to calculate SOFA change are used: paO2/fiO2, platelets, bilirubin, MAP, creatinine
#                                                                 10, 33, 20?/26, MAP=4, 19?,
# all high NaN % other than MAP, need to change 'removeRows' to not remove so many rows
print(colSequence)

noDimensions = len(colSequence)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print(device)

# # testVariable = variable.select(-2,0) #selects 0th tensor in list of tensors

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = noDimensions, int(noDimensions/2), 1, 10

#initialising training dataset
# trainingDataset = SepsisDataset(trainingUIDs[0:20000], colSequence)
# trainingDataset = SepsisDataset(trainingUIDs[:-1], colSequence)
trainingDataset = SepsisDataset(trainingUIDs[:-1], colSequence) #for testing
trainingLoader = DataLoader(dataset=trainingDataset,
                            batch_size=1, #all data from same batch must be same shape
                            #(same no rows) so either =1 or need to manipulate tensors
                            shuffle=False,
                            num_workers=1)

#initalise validation dataset
validationDataset = SepsisDataset(validationUIDs[:-1], colSequence)
validationLoader = DataLoader(dataset=validationDataset,
                            batch_size=1, #all data from same batch must be same shape
                            #(same no rows) so either =1 or need to manipulate tensors
                            shuffle=False,
                            num_workers=1)

#Initialise model
model = SepsisModel(noDimensions).to(device) #model from classes file

#Construct optimizer

# optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
#Construct the loss function
criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCELoss()
# criterion = torch.nn.KLDivLoss()

# Gradient Descent
results = []
batchMeanList, valMeanList = [], []
overfit = 0
for epoch in range(maxEpochs):
    if mode == 'train' or 'validate':
        print('epoch: ', epoch)

        #training
        print("training...")
        batchLosses = []
        for i, data in enumerate(trainingLoader, 0):
            # print("batch: " + str(i))
            x, y, UID = data

            if (x.size()[1] == 0): #empty tensor
                continue

            if (x.size()[1] == 1): #tensor with one row
                continue

            #ensure x, y tensors on correct device
            x = x.to(device)
            y = y.to(device)
            x.requires_grad_()
            y.requires_grad_()

            # Zero gradients (to prevent from retaining gradients per batch), perform a backward pass, and update the weights.
            optimizer.zero_grad() #maybe

            # Forward pass: Compute predicted y by passing x to the model
            yPred = model(x)

            # print(x)
            # print(y)
            # print(yPred)
            # Compute and print loss
            loss = criterion(yPred, y)
            # print(loss) #a single value rather than a list of values?
            # print(' loss: ', loss.item())
            # print(yPred)
            # print("batch: " + str(i) + ', loss: ', loss.item())
            #could make show file id



            # perform a backward pass (backpropagation)
            loss.backward() #calculates dx/dy
            # print(loss.grad_fn)
            # print(x.grad)

            # Update the parameters with optimiser
            optimizer.step()

            batchLosses.append(loss.item())
        batchMean = np.mean(batchLosses)
        print("--------Epoch " + str(epoch) + ", Training Loss: " + str(batchMean) + "------")
        # torch.save(model, 'model.pt') #probably should be done per file in the dataset class
    if mode == 'validate':

        # torch.load()
        # print('epoch: ', epoch)
        #validation
        with torch.no_grad():
            validationLosses = []
            #validation code
            print("validating...")
            for i, data in enumerate(validationLoader, 0):
                x, y, UID = data

                if (x.size()[1] == 0): #empty tensor
                    continue

                if (x.size()[1] == 1): #tensor with one row
                    continue

                #ensure x, y tensors on correct device
                x = x.to(device)
                y = y.to(device)
                x.requires_grad_()
                y.requires_grad_()

                model.eval()

                yPred = model(x)

                loss = criterion(yPred, y)
                #when val loss starts to increase (avg) but train loss decreases we are beginning to overfit

                #need to make it output to a file per inptut csv file
                #output file should have a row for each time where recorded
                #is risk of sepsis (real number), and a binary prediction
                #past and current rows should be used to calculate this risk
                #tSepsis, tSOFA, tSuspicion definitions may be useful
                #get values from gpu tensors
                x = x.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                yPred = yPred.cpu().detach().numpy()
                print(UID)
                if epoch == 0:
                    print(x[0], x[0].shape)
                    print(y[0], y[0].shape)

                print(yPred[0], yPred[0].shape)
                confusionMatrix(x, y, yPred)

                validationLosses.append(loss.item())
            validationMean = np.mean(validationLosses)
            print("--------Epoch " + str(epoch) + ", Validation Loss: " + str(validationMean) + "------")

        # epochTuple = (epoch, batchMean, validationMean)
        # print(epochTuple)
        # results.append(epochTuple)

        #overfitting prevention
        if epoch > 0:
            prevLocalBatchMean = np.mean(batchMeanList)
            prevLocalValMean = np.mean(valMeanList)

        if len(batchMeanList) >= 3: #only keep last three
        #thus local is mean of last three values
            batchMeanList.pop(0)
            valMeanList.pop(0)
            # print("removed")

        batchMeanList.append(batchMean)
        localBatchMean = np.mean(batchMeanList)

        valMeanList.append(validationMean)
        localValMean = np.mean(valMeanList)

        if epoch > 0:
            print("prevLocalBatchMean: " + str(prevLocalBatchMean))
            print("localBatchMean: " + str(localBatchMean))

            print("prevLocalValMean: " + str(prevLocalValMean))
            print("localValMean: " + str(localValMean))

            if prevLocalBatchMean > localBatchMean:
                if prevLocalValMean < localValMean:
                    overfit += 1
                    print("overfit+1")

        if overfit >= 2:
            print("Model has started overfitting, terminating...")
            print(results)
            #save model or save from previous?
            break

    # print(results)
