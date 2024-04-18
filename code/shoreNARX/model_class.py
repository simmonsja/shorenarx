import math

import numpy as np
import pandas as pd
import copy
import scipy
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from .models import NARX_NN

from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, FunctionTransformer
)

import tqdm

seed = 2022

################################################################################
################################################################################

class ShorelineMLModel:
    """
    A class containing the shoreline NARX model and the methods to train and 
    predict with it.

    Attributes:
        cvData: The cross-validation data used to initialize the model.
        model: The machine learning model.
        scalerX: The scaler for input data.
        scalerY: The scaler for output data.
        verbose: The verbosity level.
        settings: Additional settings for the model (see Readme.md for details)

    Methods:
        __init__: Initializes the ShorelineMLModel object.
        initialise_model: Initializes the machine learning model.
        train: Trains the machine learning model.
        predict: Makes predictions using the trained model.
        calc_performance: Calculates the performance metrics.
        fit_scalers: Fits the scalers for input and output data.
    """

    def __init__(self, cvData, **kwargs):
        """
        Initializes the ShorelineMLModel object.

        Args:
            cvData: The cross-validation data used to initialize the model.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            None
        """
        # set attrs from cvData
        self.__dict__ = cvData
        self.model = None
        self.scalerX = None
        self.scalerY = None
        self.verbose = kwargs.get('verbose', 1)
        self.settings = kwargs

    ################################################################################
    ################################################################################

    def initialise_model(self, **settings):
        """
        Initializes the model with the given settings and stores as self.model.

        Parameters:
        - settings (dict): A dictionary containing the following keys:
            - numLayers (int): The number of layers in the model.
            - layerSize (int): The size of each layer in the model.
            - dropoutRate (float): The dropout rate for the model.

        Returns:
        None
        """
        self.model = NARX_NN(self.trainX.shape[1], settings['numLayers'], settings['layerSize'], settings['dropoutRate'])
        
    ################################################################################
    ################################################################################

    def train(self, **settings):
        '''
        Create and train the model.

        Args:
            settings (dict): Dictionary containing the training settings.

        Returns:
            tuple: A tuple containing the trained model, the best model and its performance metrics, and an array of statistics.
        '''
        np.random.seed(settings.get('seed',seed))
        torch.manual_seed(settings.get('seed',seed))

        if self.model is None:
            raise ValueError('Model not initialised. Run initialise_model(**settings) first.')
        else:
            model = self.model

        #set up model/params to train
        model = model.train()
        #set loss function for training
        loss_fn = torch.nn.MSELoss()

        # define optimization algorithm
        lr = settings['learningRate']
        optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        #results store
        num = 0
        holdBest = np.inf
        bestOut = {}
        trainRMSE = np.full((int(settings['epochs']),),np.nan)
        testRMSE = np.full((int(settings['epochs']),),np.nan)
        trainNMSE = np.full((int(settings['epochs']),),np.nan)
        testNMSE = np.full((int(settings['epochs']),),np.nan)
        trainR = np.full((int(settings['epochs']),),np.nan)
        testR = np.full((int(settings['epochs']),),np.nan)
        statsEpoch = np.arange(testR.shape[0]+1)[1:]*settings['testRecordEpochs']

        #batch the data
        trainData = TensorDataset(
            torch.Tensor(self.trainX),
            torch.Tensor(self.trainY)
        )
        trainLoader = DataLoader(
            trainData, 
            batch_size=settings['batchSize']
        )
        trainAddData = TensorDataset(
            torch.Tensor(self.trainAdd),
            torch.Tensor(self.trainAdd)
        )
        trainAddLoader = DataLoader(
            trainAddData, 
            batch_size=settings['batchSize']
        )

        totalBatch = int(self.trainX.shape[0]/settings['batchSize'])
        trainLoss = np.zeros((int(settings['epochs']),))

        if settings.get('hist',True):
            repNum = self.trainX.shape[-1]
            dummyNorm = np.zeros((self.trainX.shape[1],1))
        
        torch.set_num_threads(1)
        pbar = np.arange(settings['epochs'])
        if self.verbose:
            pbar = tqdm.tqdm(pbar)
        for epoch in pbar:
            # clear stored gradient
            model.zero_grad()

            ## lr decay if any
            if settings['learningDecay'] is not None:
                lr = settings['learningRate'] * (1. / (1. + settings['learningDecay'] * epoch))
                for param_group in optimiser.param_groups:
                    param_group['learningRate'] = lr

            avg_cost = 0

            #check that the overall tensor isnt being changed
            #print(batch_stx[1][10])
            qq = 0
            for (thisbatch_x, thisbatch_y), (_, thisbatch_add) in zip(trainLoader, trainAddLoader):
                if not (~torch.isnan(thisbatch_y)).any():
                    # avoid loss step if there are nan values
                    continue
                if settings.get('hist',True):
                    prevx = thisbatch_add[0]
                    #predlist = []
                    truepred = torch.rand(thisbatch_x.shape[0])
                    for ii, thisx in enumerate(thisbatch_x):
                        newx = self.pytorch_norm_data(dummyNorm, prevx)
                        pred = model(torch.cat((thisx[:-1],torch.Tensor(newx[-1])),dim=0))
                        
                        if settings.get('dx',True):
                            pred = torch.add(pred,prevx)

                        truepred[ii] = pred

                        prevx = pred

                    # get loss
                    validinds = torch.isnan(thisbatch_y)
                    loss = loss_fn(truepred[~validinds], thisbatch_y[~validinds])
                else:
                    #doing this all together doesn't speed things up...
                    pred = model(torch.unsqueeze(thisbatch_x,0)).squeeze()
                    # get loss
                    predy = torch.add(thisbatch_add[0],pred.cumsum(0))
                    validinds = torch.isnan(thisbatch_y)
                    loss = loss_fn(predy[~validinds], thisbatch_y[~validinds])

                # perform backpropagation
                #don't let the params themselves retain gradients - backward() does this
                optimiser.zero_grad()
                # backward pass
                loss.backward()
                # update parameters
                optimiser.step()

                avg_cost += loss.item()/totalBatch
            
            if not np.mod(epoch+1,settings['testRecordEpochs']):
                model = model.eval()
                #evaluate current model on test
                testpredy = self.predict(model,
                            self.testX,
                            self.testAdd[0],
                            **settings)
                testRMSE[epoch], testNMSE[epoch], testR[epoch], _ = self.calc_performance(testpredy, self.testY)

                if testNMSE[epoch] < holdBest:
                    bestOut = {
                        'model': copy.deepcopy(model.state_dict()),
                        'num': num,
                        'epoch': epoch+1,
                        'testpredy': testpredy,
                    }
                    holdBest = testNMSE[epoch].copy()
                num += 1
                model = model.train()
                torch.manual_seed(settings.get('seed',seed))
                if self.verbose > 1:
                    print('Test NMSE: {:.2f}'.format(testNMSE[epoch]))
            if self.verbose:
                pbar.set_description('Epoch {}/{} - Loss: {:.2f}'.format(epoch+1,settings['epochs'],avg_cost))
            trainLoss[epoch] = avg_cost
        # dont bother with wasting compute if no need...
        # JS 08-2023, found that train could be replicated from state_dict but not
        # the test which implies that the model.state_dict() is not from best epoch,
        # rather I think it is from the last training epoch. So updating 
        # bestOut = {'model': model.state_dict()} to be copy.deepcopy() 
        model.load_state_dict(bestOut['model'])
        trainpredy = self.predict(
            model,
            self.trainX,
            self.trainAdd[0],
            **settings
        )

        bestOut['trainpredy'] = trainpredy
        bestOut['trainLoss'] = trainLoss
        bestOut['valLoss'] = testNMSE
        return model, bestOut, [statsEpoch,trainRMSE,trainNMSE,trainR,testRMSE,testNMSE,testR]

    ################################################################################
    ################################################################################

    def predict(self, modelsin, testX, addX, **settings):
        '''
        Predict with an existing model - this prediction is always done from the intial 
        previous shoreline position and the model predicted shoreline position is used 
        as an input for the next prediction.

        Parameters:
        - modelsin: The trained model to use for prediction.
        - testX: The input data for prediction.
        - addX: Additional input data for prediction.
        - settings: Additional settings for prediction.

        Returns:
        - truepred: The predicted output.

        '''
        # do some calcs for the stepping
        if settings.get('hist', True) or settings.get('ar1', False):
            repNum = testX.shape[-1]
            dummyNorm = np.zeros((testX.shape[1], 1))

        testX = torch.Tensor(testX)
        model = modelsin.eval()

        prevx = Variable(torch.tensor(addX).float())
        predlist = []
        num = 0
        if settings.get('hist', True) or settings.get('ar1', False):
            for thisx in testX:
                newx = self.pytorch_norm_data(dummyNorm, prevx)
                pred = model(torch.cat((thisx[:-1], torch.Tensor(newx[-1])), dim=0))

                if settings.get('dx', True):
                    pred = torch.add(pred, prevx)

                predlist.append(pred)
                prevx = pred
            truepred = torch.cat(predlist)
        else:
            pred = model(torch.unsqueeze(testX, 0)).squeeze()
            if settings.get('dx', True):
                truepred = torch.add(torch.cumsum(pred, 0).squeeze(), prevx)
            else:
                truepred = pred

        return truepred.squeeze().detach().numpy()

    ################################################################################
    ################################################################################

    def calc_performance(self,predY, obsY):
        # helper function to call actual performance calc
        return calc_performance_(predY,obsY)
     
    ################################################################################
    ################################################################################
   
    def fit_scalers(self,scTypeX=None,scTypeY=None):
        '''
        Prepare data for training
        '''
        self.scalerX = self.fit_scaler(self.trainX,scTypeX)
        self.scalerY = self.fit_scaler(self.trainY,scTypeY)

    ################################################################################
    ################################################################################

    def fit_scaler(self, fitData, sctype='stand'):
        '''
        Fits a scaler to the given data.

        Parameters:
            fitData (numpy.ndarray): The data to fit the scaler on.
            sctype (str): The type of scaler to use. Options are 'minmax', 'standard', or None.

        Returns:
            sklearn.preprocessing object: The fitted scaler object.
        '''
        if sctype == 'minmax':
            myScaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        elif sctype == 'standard':
            myScaler = StandardScaler(copy=True)
        elif sctype is None:
            myScaler = FunctionTransformer(lambda x: x,lambda x: x)
            return myScaler

        #deal with pesky y data
        if fitData.shape.__len__() == 1:
            fitData = fitData.reshape(-1,1)

        # first fit x
        myScaler.fit(fitData)

        return myScaler

    ################################################################################
    ################################################################################

    def transform_data(self,data):
        # transform data using a previously generated normaliser.
        # inputs:
        #   - data: array for X data with (r,c) = (data, input variables)
        #   - norm: scikit learn normaliser
        rnan,cnan = np.where(np.isnan(data))
        zdata = data.copy()
        zdata[rnan,cnan] = 0
        dataout = self.scalerX.transform(zdata)
        dataout[rnan,cnan] = np.nan
        return dataout
    
    ################################################################################
    ################################################################################

    def pytorch_norm_data(self, thisX, xhold):
        '''
        Normalise data and replace Day-1 in the tensor
        '''
        #replace the previous shoreline position with the normalised value
        thisX[-1] = xhold.detach().numpy()
        normold = self.scalerX.transform(thisX.T).T
        thisX[-1] = normold[-1]
        return thisX

################################################################################
################################################################################
################################################################################
################################################################################

def calc_performance_(predY, obsY):
    """
    Calculate performance metrics for predicted and observed values.

    Parameters:
    predY (array-like or DataFrame): Predicted values.
    obsY (array-like or DataFrame): Observed values.

    Returns:
    outRMSE (float): Root Mean Squared Error.
    outNMSE (float): Normalized Mean Squared Error.
    outR (float): Correlation coefficient.
    outR2 (float): Coefficient of determination.

    """
    if isinstance(predY, pd.DataFrame):
        thisPred = predY.values
        thisObs = obsY.values
    else:
        thisPred = predY
        thisObs = obsY

    nanhold = np.isnan(thisObs)
    if (thisObs[~nanhold].size == 0) or (thisPred[~nanhold].size == 0) or any(np.isnan(thisPred)):
        outRMSE, outNMSE, outR, outR2 = np.nan, np.nan, np.nan, np.nan
    else:
        outRMSE = math.sqrt(mean_squared_error(thisObs[~nanhold], thisPred[~nanhold]))
        # outR2 = r2_score(obsY[~nanhold], predY[~nanhold])
        # outR = math.sqrt(np.abs(outR2))*np.sign(outR2)
        _, _, r_value, _, _ = scipy.stats.linregress(thisObs[~nanhold], thisPred[~nanhold])
        outR = r_value
        #have to demean the data for true variance calc
        outNMSE = np.sum(np.power(thisObs[~nanhold] - thisPred[~nanhold],2))/np.sum(np.power(thisObs[~nanhold]-np.mean(thisObs[~nanhold]),2))

        # assess dx for R2
        thisdPred = np.diff(thisPred.squeeze())
        thisdObs = np.diff(thisObs.squeeze())
        nandhold = np.isnan(thisdObs)
        outR2 = r2_score(thisdObs[~nandhold], thisdPred[~nandhold])

    return outRMSE, outNMSE, outR, outR2

################################################################################
################################################################################
