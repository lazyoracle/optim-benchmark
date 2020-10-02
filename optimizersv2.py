# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:50:16 2019

@author: anurag

Quick hack to check dataset format

"""

from nevergrad.optimization import optimizerlib
from benchmark_functions import REGISTRY_OF_FUNCS as func_registry
from nevergrad.optimization import registry as algo_registry
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from datetime import date
import io
import os
import sys
from contextlib import redirect_stdout
from functools import reduce
from cycler import cycler


#Setting plot params

use_tex = False #use True if LaTeX compiler installed locally
                #Preferably use before generating the final plots since it is quite slow
plt.rcParams['figure.figsize'] = (12, 8)
plt.rc('text', usetex=use_tex) 
plt.rc('font', family='serif')
plt.rc('font', size=12)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 600 

   
def noisify(val, eps):
    """
    Add a Gaussian White noise to function value
    """ 
    
    val_noisy = (1 + random.gauss(mu=0,sigma=eps)) * (1-val)
    return val_noisy


def logify(val_noisy):
    """
    Apply Log10 to function value
    If value is negative, return the original value
    If value is greater than or equal to 1, return Log10(val)
    If 0 <= val < 1, return log10 of the max of a delta val and the original val
    MIGHT BE A SOURCE OF ARTEFACT. TO BE INVESTIGATED
    """
    
    delta = 1e-20
    if val_noisy < 0:
        return val_noisy
    elif val_noisy >= 1:
        return math.log10(val_noisy)
    else:
        return math.log10(max(delta, val_noisy))



def scalify(x):
    """
    Scale the value using arctan to [0,1]
    """
    return (np.arctan(x)+(np.pi)/2)/(np.pi)


def makeDF(arr_results, loc=0):
    """
    Make a DataFrame from the numpy array of dicts
    If part of the array of dicst has already been Dataframed,
    then start conversion from the loc position
    """
    list_for_DF = []
    count = 0
    for result in arr_results:
        if not count < loc:
            exp_data = result.pop('exp_data')
            DF_evals = pd.DataFrame(exp_data)
            result['exp_data'] = DF_evals
            result['n_evals']=len(exp_data)
            result['f_min']=exp_data[-1]['f_min']
            
        list_for_DF.append(result)
        count +=1
        
    results_DF = pd.DataFrame(list_for_DF)
    return results_DF
        




def isin_row(master, lookup, cols):
    """
    isin_row takes two datasets & a list of columns and checks if the latter is 
    a matching row in the former for the given list of columns
    Use an extension of the built-in pandas isin() function to perform a row-wise isin
    checking every row in the Dataframe for the given columns
    """
    return reduce(lambda master, lookup:master&lookup, [master[f].isin(lookup[f]) for f in cols])


def lookup(algo, func, dim, eps, log_flag, n_evals, starter, master):
    """
    Use isin_row() to check if the row formed by the passed paramters exists in the dataframe
    Chief application is to check if the current iteration in the expt has already been performed before
    DO NOT pass the full dataframe, drop the exp_data and min_params columns
    """
    check_data = pd.DataFrame({'algo': algo, 
                               'dim': dim, 
                               'func': func, 
                               'log': log_flag, 
                               'n_evals': n_evals, 
                               'noise_level': eps,
                               'starter': starter
                               }, index = [0])
    check_result = False
    if True in list(isin_row(master, check_data, cols = ['algo', 
                                                       'dim', 
                                                       'func', 
                                                       'log', 
                                                       'n_evals', 
                                                       'noise_level',
                                                       'starter'])):
        check_result = True
    return check_result


def cust_optim(optim_algo, obj_func, dim, eps, log_flag, starter, budget):
    """
    Use one of the optimizers from Nevergrad in the ask and tell interface
    Log every function call
    Return a list of dicts with the recommended min val parameters and the log of function calls
    The logified value (if done) is sent only to the optimizer, all other data logging happens with 
    the noisy value - POSSIBLE SOURCE OF ARTEFACT, CHECK REQD. DISABLED FOR NOW
    The f_min stores the current lowest attained value at the time of that evaluation number
    """
    np.random.seed(0)
    random.seed(0)
    
    optimizer = optim_algo(instrumentation=dim, budget=budget)
    evaluations_data = []
    
    f_min = 0.0
    
    initial = [starter]*dim
    
    for i in range(optimizer.budget):
        
        if i==0:
            x = initial
            dummy_x = optimizer.ask()
            params = np.asarray(x)
            final_val = scalify(obj_func(params))
            f_min = final_val
        else:
            x = optimizer.ask()
            params = np.asarray(*x.args)
            value = obj_func(params)
            noisy_val = scalify((noisify(value, eps)))
            f_min = min(f_min, noisy_val)
            final_val = noisy_val
        
#        if(log_flag):
#            final_val = logify(noisy_val)
#        else:
#            final_val = noisy_val
        
        params_candidate = optimizer.create_candidate.from_call(params)
        optimizer.tell(params_candidate, final_val)
        
        #format in which function calls get stored
        temp_evaluations_data = {'params':params,
                       'f_val':final_val,
                       'eval_no':i,
                       'f_min':f_min}
        
        evaluations_data.append(temp_evaluations_data)
        
    recommendation = optimizer.provide_recommendation()
    opt_output = {'params':recommendation,
                  'f_vals':evaluations_data
                  }
    
    return opt_output

 

def run_exp(algo_list = ['DiagonalCMA'],
            func_list = ['rosenbrock'],
            dim_list = [2, 3, 5],
            eps_list = [0.3],
            log_list = [True],
            EVAL_BUDGET = 1000,
            RESULTS_FILE = "results_df-default.pkl",
            file_type = 'pkl',
            initial_vals = [0.0],
            save_interval = 1800):
    """
    Run an expt for a given set of funcs, algos, dimensions, noise-levels etc.
    Looks up if the expt iteration has already been available in the supplied dataframe
    Saves a pickled dataframe of the logged function values along with relevant associated data
    """   
    results_list = [] #to store the data from this run of the expt
    
    try:
        #read from supplied file previous run of expts
        if file_type == 'hdf':
            stored_results = pd.read_hdf(RESULTS_FILE)
        else:
            stored_results = pd.read_pickle(RESULTS_FILE) 
        
        stripped_results = stored_results.drop(columns = ['exp_data', 'min_params']) #strip heavy section for lookup
    except:
        #any errors in file reading leads to creating a new file
        e = sys.exc_info()[0]
        print("Error in File: %s, will store expt results in new file" % e.__name__)
        lookup_flag = False
    else:
        lookup_flag = True
    

    #File for storing internal optimizer stdout 
    error_log = open('optim_error_log.txt', 'w')
    
    #Index for the runs to make it easy to index/refer to dataset
    count = 0
    
    #index for intermediate saves
    save_count = 0
    
    #run a timer to temporarily save the file every 30 mins
    save_timer = timer()
    for algo in algo_list:
        for func in func_list:
            for dim in dim_list:
                for eps in eps_list:
                    for log_flag in log_list:
                        for starter in initial_vals: 
                            if lookup_flag and lookup(algo, func, dim, eps, log_flag, EVAL_BUDGET, starter, stripped_results):
                                    #If file is read properly and lookup returns true for this run, SKIP
                                    print('[%d] - Skipping %s on %s - dim[%d] noise[%3f] log10[%s] starter[%3f]'
                                              % (count, algo, func, dim, eps, str(log_flag), starter))
                                    continue
                            else:
                                #If not run before, start new run of expt
                                start = timer()
                                print('[%d] - Running %s on %s - dim[%d] noise[%3f] log10[%s] starter[%3f] - '
                                      % (count, algo, func, dim, eps, str(log_flag), starter), end='')
                                
                                #All output from the internal optimizers is redirected to a txt
                                #file to keep the console clean
                                with redirect_stdout(error_log):
                                    try:
                                        temp_result = cust_optim(optim_algo = algo_registry[algo],
                                                             obj_func = func_registry[func],
                                                             dim = dim,
                                                             eps = eps,
                                                             log_flag = log_flag,
                                                             starter = starter,
                                                             budget = EVAL_BUDGET)
                                    except:
                                        run_error = sys.exc_info()[0]
                                        print("%s, skipping to next item" % run_error.__name__)
                                        continue
                                end = timer()
                                run_time = end - start
                                print('%3f s' %(run_time))
                                
    
                                #format of data
                                results_list.append({'algo': algo, 
                                                     'func':func,
                                                     'dim': dim, 
                                                     'noise_level':eps,
                                                     'log':log_flag,
                                                     'starter': starter,
                                                     'min_params':temp_result['params'], 
                                                     'time':run_time,
                                                     'exp_data':temp_result['f_vals']
                                                     })
                                count+=1
                                
                                
                                
                                #intermediate save
                                check_save = timer()
                                if check_save - save_timer > save_interval:
                                    
                                    print('Saving data collected so far to results_backup...')
                                    
                                    temp_arr = np.asarray(results_list) #Make a numpy array of dicts
                                    
                                    temp_df = makeDF(temp_arr, save_count-1) #Make Dataframe from array, check function makeDF() for format
                                    
                                    #reindex the dataframe before saving to file 
                                    index = [i for i in range(len(temp_df))]
                                    temp_df.index = index
                                    
                                    if file_type == 'hdf':
                                        temp_df.to_hdf('results_backup.hdf', key='df', mode='w')
                                    else:
                                        temp_df.to_pickle('results_backup.pkl')
                                                                    
                                    save_timer = timer()
                                    
                                    #location from where makeDF will start making DF in next run
                                    save_count = count+1
                                
    error_log.close()


    if not count==0:
        if save_count == (count+1):
            #no new optimisation runs after last intermediate save 
            if file_type == 'hdf': 
                os.rename('results_backup.hdf', RESULTS_FILE)
            else:
                os.rename('results_backup.pkl', RESULTS_FILE)
        else:
            results_arr = np.asarray(results_list) #Make a numpy array of dicts
            df_extend = makeDF(results_arr, save_count-1) #Make Dataframe from array, check function makeDF() for format
            
            if not lookup_flag:
                #reindex before saving
                index = [i for i in range(len(df_extend))]
                df_extend.index = index
                
                #When previous file doesn't exist or erroneous, save this run as new file            
                if file_type == 'hdf':
                    df_extend.to_hdf(RESULTS_FILE, key='df', mode='w')
                else:
                    df_extend.to_pickle(RESULTS_FILE)
            else:
                new_df = stored_results.append(df_extend) #append to dataframe read from file. NOT IN-PLACE
                
                #reindex before saving
                index = [i for i in range(len(new_df))]
                new_df.index = index
                
                if file_type == 'hdf':
                    new_df.to_hdf(RESULTS_FILE, key='df', mode='w')
                else:
                    new_df.to_pickle(RESULTS_FILE)
            


def show_algos():
    """
    List of functions presently available
    """
    return(sorted(algo_registry.keys()))
    
def show_funcs():
    """
    List of of algos presently available
    """
    return(sorted(func_registry.keys()))




def latex_safe(s):
    """
    Produce laTeX safe text
    """
    s = s.replace('_','\_')
    return s


def plot_regular(results_df, filter_func, use_tex, plot_evals = 0, y_field = 'f_min', title_prefix = '', logplot=None):
    """
    author: Shai Machnes, Anurag Saha Roy
    Plot slices from the dataframes loaded from stored expts
    """
    
    #define and set very detailed linestyles
#    linestyle_list = [(0, ()), (0, (1, 10)), (0, (1, 1)), (0, (1, 1)),
#                       (0, (5, 10)), (0, (5, 5)), (0, (5, 1)),
#                       (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
#                       (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]

    #basic linestyles
    linestyle_list = ['-', '--', '-.', ':']
    color_list = ['r', 'g', 'b', 'y']
        
    linestyle_cycler = (cycler(linestyle=linestyle_list) * 
                        cycler(color=color_list))

    plt.rc('axes', prop_cycle=linestyle_cycler)
    
    #Check to see if the entire eval budget is to be plotted, else add to title
    if plot_evals == 0:
        plot_evals = results_df['n_evals'].unique()[0]
    else:
        title_prefix = str(plot_evals) + ' evals'
    
    #apply filter function and extract the required part from the DF
    D = results_df[results_df.apply(filter_func, axis=1)]
    
    #List of field names that act as parameters for plots
    field_names = ['algo','func','dim','noise_level','log', 'starter']

    #Dividing the labels between the plot title and the plot label
    n_fields_in_title = 0
    if len(title_prefix) > 0:
        title_s = f"{title_prefix}, for " 
    else:
        title_s = ""
    
    #create labels and titles for plots
    label_format = ""
    n_fields_in_labels = 0
    for fn in field_names:
        L = list(set(D[fn]))
        if len(L) == 1:
            if n_fields_in_title > 0:
                title_s = title_s + ", "
            #Use latex_safe(fn) if text format is set to use tex
            if use_tex:
                title_s = title_s + f"{latex_safe(fn)}: {L[0]}"
            else:
                title_s = title_s + f"{fn}: {L[0]}"
            n_fields_in_title = n_fields_in_title + 1
        else:
            if n_fields_in_labels > 0:
                label_format = label_format + ", "
            #Use latex_safe(fn) if text format is set to use tex
            if use_tex:
                label_format = label_format + latex_safe(fn) + ": {d['" + fn + "']}"
            else:
                label_format = label_format + (fn) + ": {d['" + fn + "']}"
            n_fields_in_labels = n_fields_in_labels + 1
    
    
    #create pyplot figure
    fig, ((ax)) = plt.subplots(nrows=1, ncols=1)
        
    
    #Make canonical plots
    for k in range(0,len(D)):
        d = D.iloc[k]
        label_str = eval(f'f"""{label_format}"""')
        if logplot == None:
            ax.plot(d['exp_data']['eval_no'][:plot_evals+1],
                    d['exp_data'][y_field][:plot_evals+1], 
                    label=label_str)
        elif logplot == 'x':
            ax.semilogx(d['exp_data']['eval_no'][:plot_evals+1],
                    d['exp_data'][y_field][:plot_evals+1], 
                    label=label_str)
        elif logplot == 'y':
            ax.semilogy(d['exp_data']['eval_no'][:plot_evals+1],
                    d['exp_data'][y_field][:plot_evals+1], 
                    label=label_str)
        elif logplot == 'xy':
            ax.loglog(d['exp_data']['eval_no'][:plot_evals+1],
                    d['exp_data'][y_field][:plot_evals+1], 
                    label=label_str)
    
    #Plot decoration
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_xlabel(f"Evaluation No.")   
    #Use latex_safe(y_field) if text format is set to use tex
    if use_tex:
        ax.set_ylabel(latex_safe(y_field))
    else:
        ax.set_ylabel((y_field))
    fig.suptitle(title_s)
    return fig
    plt.close()


def plot_evalstoX(exp_df, shortlisted_algos, test_func, dim_list, logify_flag, noise_level, initial_vals, goalX):
    #basic linestyles
    linestyle_list = ['-', '--', '-.', ':']
    color_list = ['r', 'g', 'b', 'y']
        
    linestyle_cycler = (cycler(linestyle=linestyle_list) * 
                        cycler(color=color_list))

    plt.rc('axes', prop_cycle=linestyle_cycler)
    
    fig, ((ax)) = plt.subplots(nrows=1, ncols=1)
    
    for algo in shortlisted_algos:
        try:
            filter_func = lambda z: ((z['func'] in [test_func]) and 
                         (z['dim'] in dim_list) and 
                         (z['log'] in [logify_flag]) and
                         (z['algo'] in [algo]) and
                         (z['noise_level'] in [noise_level]) and
                         (z['starter'] in initial_vals))
            df_plot = exp_df[exp_df.apply(filter_func, axis=1)].drop(columns = ['f_min', 'min_params', 'n_evals'])
            index = [i for i in range(len(df_plot))]
            df_plot.index = index
            filter_func = lambda z: (z['exp_data'].drop(columns = ['f_val', 'params']).query('f_min < '+str(goalX)).iloc[0, 0])
            evals_to_x = df_plot.apply(filter_func, axis=1).rename('no_evals')
            df_plot = df_plot.join(evals_to_x).drop(columns = ['exp_data'])
        
            ax.loglog(df_plot['dim'], df_plot['no_evals'], label = algo)
        except:
            e = sys.exc_info()[0]
            print("Error: %s, Unable to reach %f for algo %s" % (e.__name__, goalX, algo))
            continue
    
    #Plot decoration
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_xlabel(f"No. of Dimensions")   
    ax.set_ylabel("Evaluations to "+str(goalX))
    return fig
    plt.close()


def results_summary(results_data):
    """
    Return a summary of the results stored in a dataframe.
    Returns a dict containing unique entities in each column
    Ignores the exp_data, f_min, min_params and time columns
    """
    
    unique_vals_dict = {}
    
    col_list = list(results_data.columns.values)
    for col in col_list:
        if col in ['exp_data', 'f_min', 'min_params', 'time']:
            continue
        
        uniques_temp_list = results_data[col].unique()
        unique_vals_dict[col] = list(uniques_temp_list)
    
    return unique_vals_dict


def merge_exp(list_of_exp_df):
    """
    Merge files containing data from separate runs of the expt, taking care to remove duplicates
    """
    
    #start off with first dataframe in list and add subsequent ones to it, while removing
    #any duplicate rows in the subsequent ones
    merged_df_lookup = list_of_exp_df[0].drop(columns = ['exp_data', 'min_params'])
    #list of dataframes to be concataned, after removing duplicates
    merge_df_list = [list_of_exp_df[0]]
    
    for exp_df in list_of_exp_df:
        
        #use isin_row() to check for duplicates. Pass only a stripped version of the DF for lookup
        temp_df_stripped = exp_df.drop(columns = ['exp_data', 'min_params'])
        duplicate_rows_bool_list = list(isin_row(temp_df_stripped, merged_df_lookup, cols = ['algo', 
                                                                                             'dim', 
                                                                                             'func', 
                                                                                             'log', 
                                                                                             'n_evals', 
                                                                                             'noise_level',
                                                                                             'starter']))
        
        #create a list of the rows where the new DF has duplicate data
        duplicate_rows_list = []
        for i in range(len(duplicate_rows_bool_list)):
            if duplicate_rows_bool_list[i] == True :
                duplicate_rows_list.append(i)
        
        #update merged_df_lookup for next DF 
        merged_df_lookup = merged_df_lookup.append(temp_df_stripped)
        
        #remove rows from this new DF that are duplicate
        exp_df_dropped = exp_df.drop(index = duplicate_rows_list)
        
        #reindex before finalising DF
        index = [i for i in range(len(exp_df_dropped))]
        exp_df_dropped.index = index
        merge_df_list.append(exp_df_dropped)
        
    #concatenate all the DFs in the list
    merged_df = pd.concat(merge_df_list, ignore_index=True)
    
    #sort and fix the index
    merged_df = merged_df.sort_values(by = ['algo', 'func', 'dim', 'noise_level', 'log', 'starter'], axis = 0)
    index = [i for i in range(len(merged_df))]
    merged_df.index = index
    return merged_df
    


if __name__ == "__main__":
    run_exp()