import sys, os
from joblib import Parallel, delayed

from shoreNARX import define_trial_subset


def main():
    '''
    Run the optimization process for the NARX model using Optuna.
    Run using: 
    nohup python 01a_Optuna_Run_Script.py > nohup_optuna_v02.out &
    '''
    nTrials = 25
    site = 'narra'
    version = 3
    trainingSettings = {
        'baseDir': '..',
        'site': site,
        'version': version,
        'outputDir': os.path.join('..','results','optuna'),
        'optConfig': os.path.join('..','config','optuna','{}_optuna_v{:02d}_config.json'.format(site,version)),
    }

    run_trial_subset = define_trial_subset(nTrials, trainingSettings, nStart=1)

    # This will create nProcessors x nTrials trials
    nProcessors = 4
    # with multiprocess.Pool(processes=nProcessors) as pool:
    #     pool.map(run_trial_subset, range(nProcessors))
    # There is some issue in my coding that is holding up multiprocessing
    # Rather than sort that out, I am just going to go with joblib's 
    # embarrassingly parallel loops

    # Joblib
    Parallel(n_jobs=nProcessors)(delayed(run_trial_subset)(ii) for ii in range(nProcessors))    

    print('Done!')

if __name__ == '__main__':
    main()