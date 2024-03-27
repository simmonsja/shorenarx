import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import os

################################################################################
################################################################################

def plot_CV_output(datain,stats,num,fold,cvNum=None,savefig=False):
    np.warnings.filterwarnings('ignore')

    #python indexing
    num = num - 1
    fold = fold - 1

    sns.set_context("poster")
    sns.set_style("ticks",{'axes.grid': True})

    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(111)

    data = datain[num][fold].copy()

    fillDates = pd.date_range(data['train']['modY'].index[0],data['train']['modY'].index[-1],freq=pd.infer_freq(data['train']['modY'].index[0:5]))
    trainMod = data['train']['modY'].reindex(fillDates).copy()
    #data['train']['obsY'] = data['train']['obsY'].fillna(value=np.ma.masked)

    sns.set(font_scale=2.2)
    sns.set_style("ticks")

    if any(data['train']['obsY'].index):
        ax1.errorbar(data['train']['obsY'].index,
                     data['train']['obsY'].values,
                     yerr=data['train']['stdY'].values,
                     marker='s',
                     mfc='gray',
                     mec='gray',
                     ms=5,
                     elinewidth=1,
                     fmt='.',
                     ecolor=[0.7,0.7,0.7,0.4],
                     capsize=2,
                     zorder=1,)
        trainObs = data['train']['obsY'].rolling(window=5,center=True,min_periods=3).mean().copy()
        trainObs = trainObs.reindex(fillDates)
        ax1.plot(trainObs.index,
                 np.ma.masked_where(np.isnan(trainObs.values), trainObs.values),
                 'black',
                 linewidth=1.5,
                 zorder=2,
                 label='Measured (moving mean)')
        ax1.plot(trainMod.index,
                 np.ma.masked_where(np.isnan(trainMod.values), trainMod.values),
                 sns.xkcd_rgb["red"],
                 alpha=0.9,
                 linewidth=1.5,
                 zorder=3,
                 label='Training')
    if any(data['test']['obsY'].index):
        ax1.errorbar(data['test']['obsY'].index,
                 data['test']['obsY'].values,
                 yerr=data['test']['stdY'].values,
                 marker='s',
                 mfc='gray',
                 mec='gray',
                 ms=5,
                 elinewidth=1,
                 fmt='.',
                 ecolor=[0.7,0.7,0.7,0.4],
                 capsize=2,
                 zorder=1,)
        ax1.plot(data['test']['obsY'].index,
                 data['test']['obsY'].rolling(window=5,center=True,min_periods=3).mean().values,
                 'black',
                 linewidth=1.5,
                 zorder=2)
        ax1.plot(data['test']['modY'].index,
                 data['test']['modY'].values,
                 sns.xkcd_rgb["true blue"],
                 alpha=0.9,
                 linewidth=1.5,
                 zorder=3,
                 label='Test')

    ax1.set_ylabel('Shoreline Position ($m$)')
    ax1.set_xlabel('Date')
    if num < 25:
        ax1.set_ylim([17.5, 92.5])
        ax1.set_xlim(pd.Timestamp('2004-01-01'), pd.Timestamp('2016-01-01'))
    else:
        ax1.set_ylim([35, 85])
        #ax1.set_ylim([20, 100])
        ax1.set_xlim(pd.Timestamp('1998-01-01'), pd.Timestamp('2010-01-01'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.legend(loc=4, ncol=1, fontsize=22)
    #turn warnings back on and remove from jupyter NB
    np.warnings.filterwarnings('default')

    # calcRMSE, _, _ = calc_performance(data['test']['modY'],data['test']['obsY'])
    # plt.text(0.02,0.90,'Test RMSE: {:.2f} m'.format(calcRMSE),
    #          fontsize=18, transform=ax1.transAxes)

    if savefig:
        plt.tight_layout()
        num += 1
        fold += 1
        savePath =  os.path.join('../results/CV/figures/', 'processed', 'case_' + cvNum.__str__(),
                                 'CVtesting_run_' + num.__str__() + '_fold_' + fold.__str__() + '.jpg')
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=600)


################################################################################
################################################################################

def plot_CV_ensemble(datain,fold,ax=None,savefig=False):
    # np.warnings.filterwarnings('ignore')
    thisFoldData = [_[fold] for _ in datain]

    obsData = thisFoldData[0]
    aa = [_['train']['modY'] for _ in thisFoldData]
    modDataTrain = pd.concat([_['train']['modY'] for _ in thisFoldData],axis=1)
    modDataTrain.columns = np.arange(modDataTrain.shape[1])
    modDataTest = pd.concat([_['test']['modY'] for _ in thisFoldData],axis=1)
    modDataTest.columns = np.arange(modDataTest.shape[1])

    sns.set_context("poster")
    sns.set_style("ticks",{'axes.grid': True})

    if ax is None:
        fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    fillDates = pd.date_range(modDataTrain.index[0],modDataTrain.index[-1],freq=pd.infer_freq(modDataTrain.index[0:5]))
    modDataTrain = modDataTrain.reindex(fillDates).copy()
    #data['train']['obsY'] = data['train']['obsY'].fillna(value=np.ma.masked)

    if any(obsData['train']['obsY'].index):
        ax1.errorbar(obsData['train']['obsY'].index,
                     obsData['train']['obsY'].values.squeeze(),
                     yerr=obsData['train']['stdY'].values.squeeze(),
                    #  marker='s',
                     mfc='gray',
                     mec='gray',
                     ms=2.5,
                     elinewidth=1,
                     fmt='s',
                     ecolor=[0.7,0.7,0.7,0.4],
                     capsize=2,
                     zorder=1,)
        trainObs = obsData['train']['obsY'].rolling(window=5,center=True,min_periods=3).mean().copy()
        trainObs = trainObs.reindex(fillDates)
        ax1.plot(trainObs.index,
                 np.ma.masked_where(np.isnan(trainObs.values), trainObs.values),
                 'black',
                 linewidth=1.5,
                 zorder=2,
                 label='Measured (moving mean)')
        ax1.plot(modDataTrain.index,
                 np.ma.masked_where(np.isnan(modDataTrain.mean(axis=1).values), modDataTrain.mean(axis=1).values),
                 sns.xkcd_rgb["red"],
                 alpha=0.9,
                 linewidth=1.5,
                 zorder=3,
                 label='Training')
        # fill between uncertainty bands
        ax1.fill_between(modDataTrain.index,
                np.ma.masked_where(
                    np.isnan(modDataTrain.quantile(0.05,axis=1).values),
                    modDataTrain.quantile(0.05,axis=1).values
                ),
                np.ma.masked_where(
                    np.isnan(modDataTrain.quantile(0.95,axis=1).values),
                    modDataTrain.quantile(0.95,axis=1).values
                ),
                color=sns.xkcd_rgb["red"],
                alpha=0.3,
                zorder=4)
    if any(obsData['test']['obsY'].index):
        ax1.errorbar(obsData['test']['obsY'].index,
                 obsData['test']['obsY'].values.squeeze(),
                 yerr=obsData['test']['stdY'].values.squeeze(),
                #  marker='s',
                 mfc='gray',
                 mec='gray',
                 ms=2.5,
                 elinewidth=1,
                 fmt='s',
                 ecolor=[0.7,0.7,0.7,0.4],
                 capsize=2,
                 zorder=1,)
        ax1.plot(obsData['test']['obsY'].index,
                 obsData['test']['obsY'].rolling(window=5,center=True,min_periods=3).mean().values,
                 'black',
                 linewidth=1.5,
                 zorder=2)
        ax1.plot(modDataTest.index,
                 modDataTest.mean(axis=1).values,
                 sns.xkcd_rgb["true blue"],
                 alpha=0.9,
                 linewidth=1.5,
                 zorder=3,
                 label='Test')
        #fill between uncertainty bands
        ax1.fill_between(
            modDataTest.index,
                modDataTest.quantile(0.05,axis=1).values,
                modDataTest.quantile(0.95,axis=1).values,
                color=sns.xkcd_rgb["true blue"],
                alpha=0.3,
                zorder=4
        )

    ax1.set_ylabel('Shoreline Position ($m$)', labelpad=10)
    ax1.set_xlabel('Date', labelpad=5)
    # if num < 25:
    #     ax1.set_ylim([17.5, 92.5])
    #     ax1.set_xlim(pd.Timestamp('2004-01-01'), pd.Timestamp('2016-01-01'))
    # else:
    #     ax1.set_ylim([35, 85])
    #     #ax1.set_ylim([20, 100])
    #     ax1.set_xlim(pd.Timestamp('1998-01-01'), pd.Timestamp('2010-01-01'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.legend(loc=4, ncol=1, fontsize=22)
    #turn warnings back on and remove from jupyter NB
    # np.warnings.filterwarnings('default')

    if savefig:
        plt.tight_layout()
        num += 1
        fold += 1
        savePath =  os.path.join('../results/CV/figures/', 'processed', 'case_' + cvNum.__str__(),
                                 'CVtesting_run_' + num.__str__() + '_fold_' + fold.__str__() + '.jpg')
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=600)

################################################################################
################################################################################

def plot_CV_ensemble_compare(modelsin ,fold, ax=None, savefig=False):
    # np.warnings.filterwarnings('ignore')
    mod_names = list(modelsin.keys())
    thisFoldData = [[_[fold] for _ in this_model] for this_model in modelsin.values()]

    obsData = thisFoldData[0][0]
    aa = [_['train']['modY'] for _ in thisFoldData[0]]

    modDataTest = [pd.concat([_['test']['modY'] for _ in this_model],axis=1) for this_model in thisFoldData]
    for _ in modDataTest:
        _.columns = np.arange(_.shape[1])

    sns.set_context("poster")
    sns.set_style("ticks",{'axes.grid': True})

    if ax is None:
        fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    colors = [_ for _ in sns.color_palette('Set1')]

    if any(obsData['test']['obsY'].index):
        ax1.errorbar(obsData['test']['obsY'].index,
                 obsData['test']['obsY'].values.squeeze(),
                 yerr=obsData['test']['stdY'].values.squeeze(),
                #  marker='s',
                 mfc='gray',
                 mec='gray',
                 ms=2.5,
                 elinewidth=1,
                 fmt='s',
                 ecolor=[0.7,0.7,0.7,0.4],
                 capsize=2,
                 zorder=1,)
        ax1.plot(obsData['test']['obsY'].index,
                 obsData['test']['obsY'].rolling(window=5,center=True,min_periods=3).mean().values,
                 'black',
                 linewidth=1.5,
                 zorder=2)
        for ii, this_model in enumerate(modDataTest):
            ax1.plot(
                this_model.index,
                this_model.mean(axis=1).values,
                color=colors[ii],
                alpha=0.9,
                linewidth=1.5,
                zorder=3
            )
            #fill between uncertainty bands
            ax1.fill_between(
                this_model.index,
                this_model.quantile(0.05,axis=1).values,
                this_model.quantile(0.95,axis=1).values,
                color=colors[ii],
                alpha=0.3,
                zorder=4,
                label=mod_names[ii]
            )

    ax1.set_ylabel('Shoreline Position ($m$)', labelpad=10)
    ax1.set_xlabel('Date', labelpad=5)
    # if num < 25:
    #     ax1.set_ylim([17.5, 92.5])
    #     ax1.set_xlim(pd.Timestamp('2004-01-01'), pd.Timestamp('2016-01-01'))
    # else:
    #     ax1.set_ylim([35, 85])
    #     #ax1.set_ylim([20, 100])
    #     ax1.set_xlim(pd.Timestamp('1998-01-01'), pd.Timestamp('2010-01-01'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator((1,7)))
    plt.legend(loc=4, ncol=1, fontsize=22)
    #turn warnings back on and remove from jupyter NB
    # np.warnings.filterwarnings('default')

    if savefig:
        num += 1
        fold += 1
        savePath =  os.path.join(
            '..','figures','CV' 'compare',
            '_'.join(mod_names) + '_fold_' + fold.__str__() + '.pdf'
        )
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath)

################################################################################
################################################################################
        

################################################################################
################################################################################
