#
import glob
import os, sys
import pickle
import numpy as np
import pandas as pd
import torch

import itertools 

from .model_class import ShorelineMLModel

import optuna
from optuna.study import MaxTrialsCallback
import logging
import json

################################################################################
################################################################################
# Classes
################################################################################
################################################################################

class TrainingClass:
    '''
    Class for training the shoreline NARX NN model.
    '''
    def __init__(self, *args, **kwargs):
        # self.__dict__.update(kwargs)
        self.baseDir = kwargs.get('baseDir', '.')

        # data
        self.data = None
        self.inputX = None
        self.inputY = None
        self.cvData = None

        self.model = None

        self.site = kwargs.get('site','narra')
        
        ## data params
        # x and y sample frequency
        self.xSample = kwargs.get('xSample','6H')
        self.ySample = kwargs.get('ySample','1D')
        # input variables
        self.inputVars = ['Hsig', 'Tp', 'Wdir', 'WL_mean']
        self.addVars = ['add_shl']
        self.yStd = ['shl_std']
        self.yVars = ['Shoreline']

        self.dataReq = False
        self.peakBool = False # include mean and peak over xSample

        ## model params
        self.histBool = True # RNN
        self.ar1 = False # for a model that trains with real then runs AR1

        # train params
        # CV train split
        self.trainCombs = [[1,2,2,2,2],
                           [2,1,2,2,2],
                           [2,2,1,2,2],
                           [2,2,2,1,2],
                           [2,2,2,2,1]]
        self.testEpochs = 50

        # now if there is a configuration to be had, we will have it
        self.config = self.load_config(kwargs.get('config', {}))
        self.optConfig = self.load_config(kwargs.get('optConfig', {}))

    ################################################################################
    ################################################################################

    def load_config(self, configFile):
        ''' Load params into class '''
        # read in or copy the config
        if not isinstance(configFile,dict):
            with open(configFile) as f:
                config = json.loads(f.read())
        else:
            config = configFile
        # use json for all.
        for thisKey in config.keys():
            # let json do the parsing
            if 'Bool' in thisKey:
                config[thisKey] = bool(config[thisKey])
            # and assign if present
            if thisKey in self.__dict__:
                self.__dict__[thisKey] = config[thisKey]
        return config

    ################################################################################
    ################################################################################

    def load_data(self):
        ''' Load data into class.data '''
        dataPath = os.path.join(
            self.baseDir, 'input_data', 'processed',
            '{}_xsmpl_{}_ysmpl_{}.csv'.format(self.site,self.xSample,self.ySample)
            )
        self.data = pd.read_csv(
            dataPath,index_col=0,
            parse_dates=True,dayfirst=True
        )

    ################################################################################
    ################################################################################

    def load_cv_test(self):
        ''' Load CV data into self.cvData '''
        savePath = os.path.join(
            self.baseDir,'results','models',
            'case_' + self.config['saveClass'], 
            'CV_' + self.config['site'] + '_' + 
            self.config['saveVersion'] + '_*.pkl'
        )
        fns = glob.glob(savePath)
        self.statsOut = []
        for thisFile in fns:
            # Saving the objects:
            with open(thisFile, 'rb') as f:
                self.statsOut.append(pickle.load(f))

        # first load the data
        self.load_data()
        #gather the X and Y for this case
        self.create_XY()

        if self.site == 'tairua':
            #then reduce the tairua data back to Narra size so fair is fair.
            self.inputX = self.inputX.iloc[:self.config['recordLen']]
            self.inputY = self.inputY.iloc[:self.config['recordLen']]
        # make cvData
        self.cvData = self.makeCVdata(
            self.trainCombs[0].__len__()
        )


    ################################################################################
    ################################################################################
    
    def restore_model(self,runNum,cvFold):
        ''' Load model into self.model '''
        modelSettings = {
            'numLayers': self.config['numLayers'],
            'layerSize': self.config['sizeFac'],
            'dropoutRate': self.config['dropoutRate'],
            'learningRate': self.config['learningRate'],
            'learningDecay': self.config['learningDecay'],
            'scalerTypeX': self.config['scalerType'],
            'batchSize': self.config['batchSize'],
            'epochs':self.config['epochs'],
            'testRecordEpochs': self.testEpochs,
            'modelType': self.config.get('modelType', None),
            'hist': self.histBool,
            'ar1': self.config.get('ar1', False),
            'seed': 2022 + self.config.get('runNum',0) 
        }
        if self.dataReq:
            cvData = self.cv_ttsplit(self.trainCombs[cvFold], bSize=modelSettings['batchSize'])
        else:
            cvData = self.cv_ttsplit(self.trainCombs[cvFold])
        
        # Given data, initiate a model class with embedded model and train
        self.model  = ShorelineMLModel(cvData,**modelSettings)
        self.model.initialise_model(**modelSettings)
        self.model.model.load_state_dict(self.statsOut[runNum][cvFold]['model'])
        self.model.fit_scalers(scTypeX=modelSettings['scalerTypeX'])
        # and scale X vals
        self.model.trainX = self.model.transform_data(
            self.model.trainX)
        self.model.testX = self.model.transform_data(
            self.model.testX)

    ################################################################################
    ################################################################################

    def create_XY(self):
        ''' Create X and Y from data '''
        self.inputY = reduce_to_vars(self.data, self.yVars+self.yStd+self.addVars)
        myInputVars = self.inputVars
        if self.histBool:
            #if hist bool then include
            # NOTE we don't use this as input to the model
            # XX Could this just be dummy data?
            myInputVars.append('shl_-1_0')
        self.inputX = reduce_to_vars(self.data, myInputVars)
        if not self.peakBool:
            self.inputX = self.inputX[[_ for _ in self.inputX.columns if 'peak' not in _]]

    ################################################################################
    ################################################################################

    def makeCVdata(self,splitNum):
        # split all data into folds

        # Inputs:
        #   mlData     : short term model component data
        #   vars       : dict containing - 'inputVars', 'addVars' and 'yVars'
        #   split      : split decimal
        #
        # Outputs:
        #   out        : dict object containing the split train and test data as X/Y
        cvSize = int(np.floor(self.inputY.shape[0]/splitNum))

        out = []
        for ii in range(splitNum):
            baseNum = cvSize*ii

            #collect data for each fold
            thisData = {
                'X': self.inputX.iloc[baseNum:baseNum+cvSize,:].values,
                'Y': self.inputY[self.yVars[0]].iloc[baseNum:baseNum+cvSize].values.squeeze(),
                'Add': self.inputY[self.addVars[0]].iloc[baseNum:baseNum+cvSize].values.squeeze(),
            }
            #add y std data if available
            if self.yStd:
                thisData['Y_std'] = self.inputY[self.yStd[0]].iloc[baseNum:baseNum+cvSize].values.squeeze()

            thisData['Index'] = self.inputY.iloc[baseNum:baseNum+cvSize].index
            out.append(thisData)
        return out


    ################################################################################
    ################################################################################
   
    def cv_Instance(self,ttComb, settings={}):
        if self.dataReq:
            cvData = self.cv_ttsplit(ttComb, bSize=settings['batchSize'])
        else:
            cvData = self.cv_ttsplit(ttComb)
        
        # Given data, initiate a model class with embedded model and train
        shorelineModel = ShorelineMLModel(cvData,**settings)
        
        #normalise
        shorelineModel.fit_scalers(scTypeX=settings['scalerTypeX'])
        shorelineModel.trainX = shorelineModel.transform_data(
            cvData['trainX'])
        shorelineModel.testX = shorelineModel.transform_data(
            cvData['testX'])

        # train the model
        shorelineModel.initialise_model(**settings)
        model, bestEpoch, trainStats = shorelineModel.train(**settings)
        
        df_testingdata = {
            'test': {
                'obsY': pd.DataFrame(data=cvData['testY'],
                                    index=pd.to_datetime(cvData['testIndex'])),
                'modY': pd.DataFrame(data=bestEpoch['testpredy'],
                                    index=pd.to_datetime(cvData['testIndex'])),
                'stdY': pd.DataFrame(data=cvData['testY_std'],
                                    index=pd.to_datetime(cvData['testIndex'])),
            },
            'train': {
                'obsY': pd.DataFrame(data=cvData['trainY'],
                                    index=pd.to_datetime(cvData['trainIndex'])),
                'modY': pd.DataFrame(data=bestEpoch['trainpredy'],
                                    index=pd.to_datetime(cvData['trainIndex'])),
                'stdY': pd.DataFrame(data=cvData['trainY_std'],
                                    index=pd.to_datetime(cvData['trainIndex'])),
            },
            'norm_converter': shorelineModel.scalerX,
            'ttcomb': ttComb,
            'model': bestEpoch['model'],
            'num': bestEpoch['num'],
            'epochs': trainStats[0],
            'trainRMSE': trainStats[1],
            'trainNMSE': trainStats[2],
            'trainR': trainStats[3],
            'testRMSE': trainStats[4],
            'testNMSE': trainStats[5],
            'testR': trainStats[6],
        }

        return df_testingdata

    ################################################################################
    ################################################################################

    def cv_Merge(self,listin):
        # this function concatenates data for input with arbitrary keys
        out = {}
        for mykey in listin[0].keys():
            out[mykey] = np.concatenate([_[mykey] for _ in listin])
        return out

    ################################################################################
    ################################################################################

    def cv_ttsplit(self, ttbool, bSize=0):
        # this function concatenates data for input with arbitrary keys and names
        # correctly.
        # input:
        #   - datain = a list of dicts with train/test data
        #   - ttbool = 1 where test data, 2 where train data

        #compress and merge the data
        trainData = list(itertools.compress(self.cvData, [_ == 2 for _ in ttbool]))
        if bSize > 0:
            trainDataNew = []
            for thisTD in trainData:
                thisSize = thisTD['X'].shape[0]
                if np.remainder(thisSize,bSize):
                    extendLen = int(np.ceil(np.divide(thisSize,bSize)) * bSize - thisSize)
                    for key in thisTD.keys():
                        if 'Y' in key:
                            thisTD[key] = np.concatenate([thisTD[key],np.repeat(np.expand_dims(np.nan,0),extendLen,axis=0)],axis=0)
                        else:
                            thisTD[key] = np.concatenate([thisTD[key],np.repeat(np.expand_dims(thisTD[key][-1],0),extendLen,axis=0)],axis=0)
                trainDataNew.append(thisTD)
        else:
            trainDataNew = trainData

        train_data = self.cv_Merge(trainDataNew)
        if any([_ == 1 for _ in ttbool]):
            test_data = self.cv_Merge(list(itertools.compress(self.cvData, [_ == 1 for _ in ttbool])))
        else:
            test_data = {}
        #now name properly
        data = {}
        for thisCase, mergeData in zip(['train','test'],[train_data,test_data]):
            for thisKey in mergeData.keys():
                data[thisCase + thisKey] = mergeData[thisKey]
        return data

    ################################################################################
    ################################################################################
 
    def trainCombsGenerate(num,valNum,rs=1):
        nck = int(num/valNum)
        #nck = int(sp.special.comb(num,chooseNum)/(num/chooseNum))
        keepNum = np.arange(5,81,5)
        # how many of the cv divisions
        keepInt = [int(_) for _ in num*keepNum/100]
        listLen = int(nck * keepInt.__len__())

        cntr = 0
        combs = np.full((listLen,num),2)
        for jj in np.arange(nck):
            for ii, kk in enumerate(keepInt):
                # set the validation sets
                # need to change this to be all of one cv fold then all of another
                combs[cntr,jj*valNum:jj*valNum+valNum] = 1
                #combs[ii*nck+jj,jj+1] = 1
                freeInds = [_ for _ in np.arange(num) if not _ in np.arange(jj*valNum,jj*valNum+valNum)]

                # get a random sample and vary a bit but not conflicting with others
                np.random.seed(rs+50*cntr)
                dropInds = np.random.choice(freeInds,replace=False,size=num-valNum-kk)
                combs[cntr,dropInds] = 0
                cntr += 1
        np.random.seed(rs)
        return combs

    ################################################################################
    ################################################################################
    # Training end function

    def training_wrapper(self, saveBool=False):
        # first load the data
        self.load_data()
        #gather the X and Y for this case
        self.create_XY()

        if self.site == 'tairua':
            #then reduce the tairua data back to Narra size so fair is fair.
            self.inputX = self.inputX.iloc[:self.config['recordLen']]
            self.inputY = self.inputY.iloc[:self.config['recordLen']]
        
        self.cvData = self.makeCVdata(
            self.trainCombs[0].__len__()
        )

        modelSettings = {
            'numLayers': self.config['numLayers'],
            'layerSize': self.config['sizeFac'],
            'dropoutRate': self.config['dropoutRate'],
            'learningRate': self.config['learningRate'],
            'learningDecay': self.config['learningDecay'],
            'scalerTypeX': self.config['scalerType'],
            'batchSize': self.config['batchSize'],
            'epochs':self.config['epochs'],
            'testRecordEpochs': self.testEpochs,
            'verbose': self.config.get('verbose', True),
            'hist': self.histBool,
            'ar1': self.config.get('ar1', False),
            'seed': 2022 + self.config.get('runNum',0) 
        }

        # now run through the cv_Instance
        statsOut = []
        for thisComb in self.trainCombs:
            statsOut.append(self.cv_Instance(thisComb, settings=modelSettings))

        if saveBool:
            savePath = os.path.join(
                self.baseDir,'results','models',
                'case_' + self.config['saveClass'], 
                'CV_{}_{}_{}.pkl'.format(
                    self.config['site'],
                    self.config['saveVersion'],
                    self.config.get('runNum',0)
                )
            )
            os.makedirs(os.path.dirname(savePath),exist_ok=True)
            # Saving the objects:
            with open(savePath, 'wb') as f:
                pickle.dump(statsOut, f)

        return np.nanmean([np.nanmin(_['testRMSE']) for _ in statsOut]) 

    ################################################################################
    ################################################################################
    # Grid Search

    def gs_objective(self,trial):
        # Optimisation Goalss
        # data params
        xSample_ = self.optConfig.get('xSample',['3H','6H','12H','1D'])
        if isinstance(xSample_,list):
            self.xSample = trial.suggest_categorical(
                "x_sample", 
                xSample_
            )
        else:
            self.xSample = xSample_
        ySample_ = self.optConfig.get('ySample',['1D','3D'])
        if isinstance(ySample_,list):
            self.ySample = trial.suggest_categorical(
                "y_sample", 
                ySample_
            )
        else:
            self.ySample = ySample_
        peakVar_ = self.optConfig.get('peakVar',[True,False])
        if isinstance(peakVar_,list):
            self.peakVar = trial.suggest_categorical(
                "peak_variable",
                peakVar_
            )
        else:
            self.peakVar = peakVar_
        
        # Network architecture
        numLayers_ = self.optConfig.get('numLayers',[1,2])
        if isinstance(numLayers_,list):
            numLayers = trial.suggest_int("num_layers", numLayers_[0], numLayers_[1])
        else:
            numLayers = numLayers_
        sizeFac_ = self.optConfig.get('sizeFac',[10,710,50])
        if isinstance(sizeFac_,list):
            sizeFac = trial.suggest_int("size_factor", sizeFac_[0], sizeFac_[1], step=sizeFac_[2])
        else:
            sizeFac = sizeFac_
        # Dropout param
        dropoutRate_ = self.optConfig.get('dropoutRate',[0.0,0.4])
        if isinstance(dropoutRate_,list):
            dropoutRate = trial.suggest_float(
                "dropout_rate", dropoutRate_[0], dropoutRate_[1]
            )
        else:
            dropoutRate = dropoutRate_
        # Learning rate param 
        learningRate_ = self.optConfig.get('learningRate',[1e-5, 1e-3])
        if isinstance(learningRate_,list):
            learningRate = trial.suggest_float(
                "learning_rate", learningRate_[0], learningRate_[1], 
                log=True
            )
        else:
            learningRate = learningRate_
        learningDecay_ = self.optConfig.get('learningDecay',[1e-3, 1e-1])
        if isinstance(learningDecay_,list):
            learningDecay = trial.suggest_float(
                "learning_decay", learningDecay_[0], learningDecay_[1]
            )
        else:
            learningDecay = learningDecay_
        scalerType_ = self.optConfig.get('scalerType',['standard','minmax'])
        if isinstance(scalerType_,list):
            scalerType = trial.suggest_categorical(
                "scaler", 
                scalerType_
            )
        else:
            scalerType = scalerType_
        batchSize_ = self.optConfig.get('batchSize',[4,20,4])
        if isinstance(batchSize_,list):
            batchSize = trial.suggest_int(
                "batch_size", 
                batchSize_[0], batchSize_[1], step=batchSize_[2]
            )
        else:
            batchSize = batchSize_
        epochs_ = self.optConfig.get('epochs',[1000,4000,1000])
        if isinstance(epochs_,list):
            epochs = trial.suggest_int(
                "epoch_num",
                epochs_[0], epochs_[1], step=epochs_[2]
            )
        else:
            epochs = epochs_

        # first load the data
        self.load_data()
        #gather the X and Y for this case
        self.create_XY()

        if self.site == 'tairua':
            #then reduce the tairua data back to Narra size so fair is fair.
            self.inputX = self.inputX.iloc[:self.config['recordLen']]
            self.inputY = self.inputY.iloc[:self.config['recordLen']]
        
        self.cvData = self.makeCVdata(
            self.trainCombs[0].__len__()
        )

        modelSettings = {
            'numLayers': numLayers,
            'layerSize': sizeFac,
            'dropoutRate': dropoutRate,
            'learningRate': learningRate,
            'learningDecay': learningDecay,
            'scalerTypeX': scalerType,
            'batchSize': batchSize,
            'epochs': epochs,
            'testRecordEpochs': self.testEpochs,
            'verbose': False,
            'hist': self.histBool
        }

        # now run through the cv_Instance
        statsOut = []
        for thisComb in self.trainCombs:
            statsOut.append(self.cv_Instance(thisComb, settings=modelSettings))
         
        return np.nanmean([np.nanmin(_['testRMSE']) for _ in statsOut])

    ################################################################################
    ################################################################################



################################################################################
################################################################################
#Functions
################################################################################
################################################################################

# Settings 
def define_trial_subset(nTrials,trainingSettings,nStart=0):
    '''
    Inputs:
        nTrials: number of trials to run
        trainingSettings: dictionary of training settings
    '''
    def run_trial_subset(num):
        studyName = os.path.join(
            trainingSettings['outputDir'],
            'study_{:02.0f}_{}_v{:02.0f}'.format(num+nStart,trainingSettings['site'],trainingSettings['version'])
        )

        trainObj = TrainingClass(**trainingSettings)

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('{}.log'.format(studyName,mode='w')))
        study = optuna.create_study(
            study_name=studyName,
            # pruner=optuna.pruners.MedianPruner(),
            storage = 'sqlite:///{}.db'.format(studyName),
            load_if_exists=True,
        )
        study.optimize(
            trainObj.gs_objective,
            n_trials=nTrials,
            callbacks=[MaxTrialsCallback(nTrials)],#,states=optuna.trial.TrialState.COMPLETE)],
        )
    return run_trial_subset

################################################################################
################################################################################

# Settings 
def define_cv_run(nTrials,trainingSettings,nStart=0):
    '''
    Inputs:
        nTrials: number of trials to run
        trainingSettings: dictionary of training settings
    '''
    def run_trial_subset(num):
        for ii in np.arange(nTrials):
            printNum =  nStart + num*nTrials+ii
            print('Beginning Run {}'.format(printNum))
            trainObj = TrainingClass(**trainingSettings)
            # train as per config
            trainObj.config['runNum'] = printNum
            trainObj.config['verbose'] = False
            # train as per config
            res = trainObj.training_wrapper(saveBool=True)
            print('Run {} finished - {:.2f}'.format(printNum,res))
    return run_trial_subset

################################################################################
################################################################################

def reduce_to_vars(inData,vars):
    boolsOut = np.full(inData.columns.shape, False)
    for thisVar in vars:
        thisBools = [ii for ii, _ in enumerate(inData.columns.tolist()) if _.startswith(thisVar)]
        boolsOut[thisBools] = True
    return inData[inData.columns[boolsOut]]

################################################################################
################################################################################
