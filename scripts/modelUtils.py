# basic imports
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, List

# basic plotting library
import matplotlib.pyplot as plt

# DL library imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from  torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR

# metrics calculation
from sklearn.metrics import accuracy_score


def setSeed(seed : int):
    """Function to make results reproducible
    Args:
        seed (int): input seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def printModelSparsity(model : nn.Module, modelName : str):
    """Function prints sparsity of the model - the percentage of weights with 0.0 in each layer of the model
    Args:
        model (nn.Module): input model
        modelName (str): name of the models
    """    
    
    print("------------------------")    
    print(f'{modelName} sparsity Information')    
    print("------------------------")
    totalElements = 0
    totalZeroElements = 0 
    for name, module in model.named_children():
        numElementsInLayer = float(module.weight.nelement())
        numZeroElementsInLayer = float(torch.sum(module.weight == 0))
        layerWeightSparsity = 100.0 * (numZeroElementsInLayer/ numElementsInLayer)
        print(f"Sparsity in {name}.weight: {layerWeightSparsity}%")
        
        totalElements += numElementsInLayer
        totalZeroElements += numZeroElementsInLayer
    
    if totalElements > 0 :
        globalWeightSparsity = 100.0 * (totalZeroElements / totalElements)
    else:
        globalWeightSparsity = 0.0
    print(f"Global Sparsity (weight): {globalWeightSparsity}%")


def plotTrainingResults(df:pd.DataFrame, modelName:str):
    """Function plots training, validation losses vs epoch
    Args:
        df (pd.DataFrame): Dataframe with the columns ['epoch', 'trainLoss', 'validationLoss']
        modelName (str): name of the model
    """   
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1_color = 'tab:red'
    ax1.set_ylabel('trainLoss', color=ax1_color)
    ax1.plot(df['epoch'].values, df['trainLoss'].values, color=ax1_color)
    ax1.tick_params(axis='y', labelcolor=ax1_color)

    ax2 = ax1.twinx()  
    ax2_color = 'tab:blue'    
    ax2.set_ylabel('validationLoss', color=ax2_color)
    ax2.plot(df['epoch'].values, df['validationLoss'].values, color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)

    fig.tight_layout()
    plt.show()



class meanClassificationAccuracyMetric:
    """
    Class to find the avg accuracy of softmax predictions to ground truth label
        CFG (Any): object containing num_classes 
        device (torch.device): compute device
    """    
    def __init__(self, CFG, device):
        self.batchAccuracies = []
        self.meanClassificationAccuracy = 0.0

    def update(self, y_preds: torch.Tensor, labels: torch.Tensor):
        """ Function finds the classification accuracy for the input batch

        Args:
            y_preds (torch.Tensor): model predictions
            labels (torch.Tensor): groundtruth labels        
        Returns
        """
        # predicted output class
        modelPredictions = np.argmax(torch.softmax(y_preds, axis=1).numpy(), axis=1)
        self.batchAccuracies.append(accuracy_score(labels.numpy(), modelPredictions))

    def compute(self):
        """ returns meanClassificationAccuracy """
        self.meanClassificationAccuracy = np.mean(self.batchAccuracies)
        return self.meanClassificationAccuracy

    def reset(self):
        self.batchAccuracies = []
        self.meanClassificationAccuracy = 0.0






def trainValidateModel_KD(studentModel:nn.Module, criterion, distillationLossFn,
                        optimizer, dataloader_train:DataLoader, dataloader_valid:DataLoader, metricFunction, 
                        metricName :str, device:torch.device, CFG:Any, modelName:str, lr_scheduler=None, 
                        saveModel : bool =False, verbose : bool=False, plotResults : bool=False):
    """Function runs train and validation cycles of the Student model on given dataset using
    teacher model and inputs in `CFG` class. 
    - The teacher model is assumed to have been loaded with pretrained weights
    and only the student model is trained. 
    - CFG must contain `T` and `alpha` parameters required for knowledge distillation

    Args:
        studentModel (nn.Module): model, which tries to distill knowledge from teacher 
        criterion ([type]): loss function between student predictions and actual labels (hard targets)
        distillationLossFn ([type]): function to calcualted distilled loss 
        optimizer ([type]): student model optimizer such as Adam, SGD etc
        dataloader_train (DataLoader): train set 
        dataloader_valid (DataLoader): validation set
        metricFunction ([type]): function that calculates metric b/w predicted and ground truth  
        metricName (str) : name of the metric
        device (torch.device): compute device as CPU, GPU
        CFG (Any): class containing info on num epochs, learning rate etc
        modelName (str): name of the model
        lr_scheduler ([type], optional): [description]. Defaults to None.
        saveModel (bool, optional): [description]. Defaults to False.
        verbose (bool, optional): [description]. Defaults to False.
        plotResults (bool, optional): [description]. Defaults to False.
    Returns:
        [type]: [description]
    """    

    print("------------------------")
    print(f"Train Validate Pipeline for - {modelName} on {str(device)}")
    print("------------------------")
    results = []    
    minValidationLoss = np.Inf
    lenTrainLoader = len(dataloader_train)
    scaler = GradScaler()
    
    # move student model to target device
    studentModel.to(device)

    # read knowledge distillation parameters 
    try:
        alpha = CFG.alpha
    except:
        alpha = 0.5
    
    try:
        temperature = CFG.T
    except:
        temperature = 1.0

    print("------------------------")
    print(f"KD parameters = - alpha = {alpha}, temperature = {temperature}")
    print("------------------------")


    for epoch in range(CFG.N_EPOCHS):
        if verbose == True:
            print(f"Starting {epoch + 1} epoch ...")
        
        # Training
        studentModel.train()
        trainLoss = 0.0
        for i, (inputs, labels, teacherModelPreds) in tqdm(enumerate(dataloader_train), total=lenTrainLoader):
            inputs = inputs.to(device).float()
            labels = labels.to(device)  

            # builtin package to handle automatic mixed precision            
            with autocast():
                # Forward pass on teacher and student model
                studentModelPreds = studentModel(inputs)

                # total loss = hard target loss fn + distillation loss using teacher predictions
                loss = distillationLossFn(labels, studentModelPreds, teacherModelPreds, 
                                        criterion, temperature, alpha)
                trainLoss += loss.item()
                    
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
            
            # adjust learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, (OneCycleLR, CosineAnnealingLR, LambdaLR)):    
                    lr_scheduler.step()
                if isinstance(lr_scheduler, (CosineAnnealingWarmRestarts)):    
                    lr_scheduler.step(epoch + i / lenTrainLoader)

        
        # Validate
        studentModel.eval()
        validationLoss = 0.0
        metricObject = metricFunction(CFG, device)


        with torch.no_grad():
            for inputs, labels, _ in dataloader_valid:
                inputs = inputs.to(device)
                labels = labels.to(device)  
                y_preds = studentModel(inputs)
            
                # calculate loss
                loss = criterion(y_preds, labels)
                validationLoss += loss.item()
                
                # update batch metric information            
                metricObject.update(y_preds.cpu().detach(), labels.cpu().detach())

        # compute per batch losses
        trainLoss = trainLoss / len(dataloader_train)
        validationLoss = validationLoss / len(dataloader_valid)

        # compute metric
        validationMetric = metricObject.compute()

        if verbose == True:
            print(f'Epoch: {epoch+1}, trainLoss:{trainLoss:6.5f}, validationLoss:{validationLoss:6.5f}, {metricName}:{validationMetric: 4.2f}%')
        
        # store results
        results.append({'epoch': epoch, 'trainLoss': trainLoss, 
                        'validationLoss': validationLoss, f'{metricName}': validationMetric})
        
        # if validation loss has decreased and user wants to
        if validationLoss <= minValidationLoss:
            minValidationLoss = validationLoss
            if saveModel == True:
                torch.save(studentModel.state_dict(), f'{modelName}.pt')


    if plotResults ==True:
        results = pd.DataFrame(results)
        plotTrainingResults(results, f'{modelName}')                
    return results





def trainValidateModel(model:nn.Module, criterion, optimizer, dataloader_train : DataLoader, 
                       dataloader_valid : DataLoader, metricFunction, metricName :str,  device : torch.device, CFG : Any, modelName:str, 
                       lr_scheduler=None, saveModel : bool =False, verbose : bool=False, plotResults : bool=False):
    """Function runs train and validation cycles of given model on given datasets according
    to inputs in `CFG` class

    Args:
        model (nn.Module): input model
        criterion ([type]): 
        optimizer ([type]): optimizer function, eg: Adam, SGD
        dataloader_train (DataLoader): train set 
        dataloader_valid (DataLoader): validation set
        metricFunction ([type]) : function that calculates metric b/w predicted and ground truth  
        metricName (str) : name of the metric
        device (torch.device): compute device as CPU, GPU
        CFG (Any): class containing info on num epochs, learning rate etc
        modelName (str): name of the model
        saveModel (bool, optional): [description]. Defaults to False.
        verbose (bool, optional): [description]. Defaults to False.
        plotResults (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    print("------------------------")
    print(f"Train Validate Pipeline for - {modelName} on {str(device)}")
    print("------------------------")
    results = []    
    minValidationLoss = np.Inf
    lenTrainLoader = len(dataloader_train)
    scaler = GradScaler()

    # move model to target device
    model.to(device)
    
    for epoch in range(CFG.N_EPOCHS):
        if verbose == True:
            print(f"Starting {epoch + 1} epoch ...")
        
        # Training
        model.train()
        trainLoss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(dataloader_train), total=lenTrainLoader):
            inputs = inputs.to(device)
            labels = labels.to(device)  

            # builtin package to handle automatic mixed precision            
            with autocast():
                # Forward pass
                y_preds = model(inputs.float())
                loss = criterion(y_preds, labels)
                trainLoss += loss.item()
                    
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
            
            # adjust learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, (OneCycleLR, CosineAnnealingLR, LambdaLR)):    
                    lr_scheduler.step()
                if isinstance(lr_scheduler, (CosineAnnealingWarmRestarts)):    
                    lr_scheduler.step(epoch + i / lenTrainLoader)

        # Validate
        model.eval()
        validationLoss = 0.0
        metricObject = metricFunction(CFG, device)

        with torch.no_grad():
            for inputs, labels in dataloader_valid:
                inputs = inputs.to(device)
                labels = labels.to(device)                
                y_preds = model(inputs)
            
                # calculate loss
                loss = criterion(y_preds, labels)
                validationLoss += loss.item()

                # update batch metric information            
                metricObject.update(y_preds.cpu().detach(), labels.cpu().detach())

        # compute per batch losses
        trainLoss = trainLoss / len(dataloader_train)
        validationLoss = validationLoss / len(dataloader_valid)

        # compute metric
        validationMetric = metricObject.compute()

        if verbose == True:
            print(f'Epoch: {epoch+1}, trainLoss:{trainLoss:6.5f}, validationLoss:{validationLoss:6.5f}, {metricName}:{validationMetric: 4.2f}%')
        
        # store results
        results.append({'epoch': epoch, 'trainLoss': trainLoss, 
                        'validationLoss': validationLoss, f'{metricName}': validationMetric})
        
        # if validation loss has decreased and user wants to
        if validationLoss <= minValidationLoss:
            minValidationLoss = validationLoss
            if saveModel == True:
                torch.save(model.state_dict(), f'{modelName}.pt')
                # torch.jit.save(torch.jit.script(model), f'{CFG.PRETRAINED_PATH}')

    if plotResults ==True:
        results = pd.DataFrame(results)
        plotTrainingResults(results, f'{modelName}')                
    return results


def evaluteOnTestData(model :nn.Module, pretrainedModelPath:str, device :torch.device, 
                     dataloader_test : DataLoader, metricFunction, metricName : str,
                     modelName:str, CFG:Any, verbose : bool =False) ->float:
    """Evaluate the model on test set

    Args:
        model (nn.Module): input model
        pretrainedModelPath (str): path of weight file
        device (torch.device): 
        dataloader_test (DataLoader): test dataset
        metricFunction () : 
        metricName (str) : name of metric
        modelName (str): name of the model
        CFG (Any) : object containing num_classes
        verbose (bool, optional): flag to print results. Defaults to False.

    Returns:
        testSetMetric(float): 

    Reference:
        https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
    """
    testSetMetric = 0.0

    if verbose == True:
        print("------------------------")
        print(f"Test Data Results for {modelName} using {str(device)}")
        print("------------------------")
    
    modelLoadStatus = False
    if pretrainedModelPath is not None:
        if os.path.isfile(pretrainedModelPath) == True:
            model.load_state_dict(torch.load(pretrainedModelPath, map_location=device))
            modelLoadStatus = True
    # no need to load model
    else:
        modelLoadStatus = True

    if modelLoadStatus == True:
        lenTestLoader = len(dataloader_test)
        model.to(device)
        # set to inference mode
        model.eval()
        metricObject = metricFunction(CFG, device)

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader_test, total=lenTestLoader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                y_preds = model(inputs)
            
                # update batch metric information            
                metricObject.update(y_preds.cpu().detach(), labels.cpu().detach())

        # compute metric of test set predictions
        testSetMetric = metricObject.compute()
        
        if verbose == True:
            print(f'{modelName} has {testSetMetric} {metricName} on testData')
    else:
        print(f'Model cannot load state_dict from {pretrainedModelPath}')
    return testSetMetric