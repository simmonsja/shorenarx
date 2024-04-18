import os
from joblib import Parallel, delayed

from shoreNARX import define_trial_subset


def main():
    '''
    Run the optimization process for the NARX model using Optuna. The configuration of the optuna study is defined in the config file (in config/optuna/ directory) which shows fixed and optimisable parameters. The results are saved in the results/optuna/ directory.

    Run from the code/ directory using e.g.: 
    nohup python 01_Optuna_Run_Script.py > nohup_optuna_v01.out &

    '''
    nTrials = 25
    site = 'narra' # 'narra' or 'tairua'
    version = 3
    trainingSettings = {
        'baseDir': '..',
        'site': site,
        'version': version,
        'outputDir': os.path.join('..','results','optuna'), # Directory to save the results
        'optConfig': os.path.join('..','config','optuna','{}_optuna_v{:02d}_config.json'.format(site,version)), # Configuration file for the optuna study
    }

    run_trial_subset = define_trial_subset(nTrials, trainingSettings, nStart=1)

    # Number of processors for parallel jobs, this will create nProcessors x nTrials trials overall
    nProcessors = 4

    # Joblib
    Parallel(n_jobs=nProcessors)(delayed(run_trial_subset)(ii) for ii in range(nProcessors))    

    print('Done!')

if __name__ == '__main__':
    main()