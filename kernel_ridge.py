#from types import NoneType
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np
import utils_functions as utils
import utils_functions_for_validation as utils_for_validation
import pandas as pandas
import time
import os

import computing_marginal_effects as comput_margin

############# using statsmodels API
import statsmodels
import statsmodels.api as sm
import statsmodels.sandbox as sm_sandbox
from statsmodels.sandbox.regression.kernridgeregress_class import GaussProcess
import sqlite3
# WARNING !!!!! THE VALUE OF ENERGY EFFICIENCY USED BY THE MODEL IS OBTAINED WITH FORMULA WORKLOAD/ENERGY, 
#    IT IS THE INVERSE OF THE ONE COMPUTED BY THE AUTOMATIZATION SCRIPT
########### General option on input datas
phone_name = "google_pixel_4a_5g" #  can be "google_pixel_4a_5g",  or "samsung_galaxy_s8"
input_format = "google_pixel_4a_5g_format"  # "google_pixel_4a_5g_format" # can be "google_pixel_4a_5g_format", "samsung_galaxy_s8_format", or "generic"
base_Y__X_meaning_dictionnary, base_Y_N_on_socket__X_meaning_dictionnary, base_Y_F__X_meaning_dictionnary, base_Y_F_N_on_socket__X_meaning_dictionnary = utils.setup_X_format_meaning_dictionnaries(phone_name)

energy_gap = 10  # in mAh, this parameter and the next one is used in the function utils.remove_aberrants_points to remove some "aberrant_point"
                # where the energy measured on a configuration is not in the correct interval regarding "similar" configuration
number_of_neighbour = 10  # number of "similar" neighbours to consider

max_input_size = -1
convert_X_to_base_Y = True
X_format_in_model = "base_Y"# can be - base_Y (for the base representing limitation on some phone) 
                            # - or base_Y_N_on_socket with the previous base, but only the number of cores on little socket has been retained,
                            # not the state of every core as on base_Y 
                            # - or base_Y_F, where the frequency of the socket can be unmodify, for each socket a variable is added to save this information (frequency is free or Not)
                            # - or base_Y_F_N_on_socket, similar to base_Y_N_on_socket, except the fact the socket frequency can be unmodified

consider_automatization_summaries = True 

automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/summary_files_only"
#automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/can_be_reused/GOOGLE_PIXEL_RESULTS/summary_files_only_0.89"


consider_exact_values_of_frequency =  False # considerer exact values of frequencies of O and 1
populate_inputs_to_considere_thread_combinations_on_same_socket = False
maximum_number_of_combination = 1000 # Maximum number of combinaitions per configuration when populating X
one_hot_encoding_of_frequency = False # if we do not consider_exact_values_of_frequency, do we explode frequency encoding , 0,1,1, means midle
                                      # this procedure should be done after populating X if needed
standartize_inputs = False
value_to_retain = "median" # or "mean". When removing duplicates, if the corresponding energy efficiency of each duplicates are different each other,
                           # we retained by default the median efficiency, maybe in some cases the retaining the "mean" can be usefull. 
remove_aberrant_points = True 
remove_duplicates = True
fill_data_from_folders = True
########### General options of the model
alpha = 0.01000000099 # [ best value for Base_Y format 0.01000000099 ] #0.44930060152149415 #0.730902889668161 # 1e-4
search_ridge_coeff = False  # MODIFIED before integrating experiment automatization
search_strategy =  "explore_all_values"      # "dichotomic" # can be sequential, dichotomic, explore_all_values
#ltolerance =  9415743.8 #1000000 #9415743.7# 9415743.7 #0.001                           9415743.7 -> R2 =  0.39128379283199
ltolerance = 1e+9 # with exact units of mesurement (workload and energy), the energy efficiency is aroung 1e-11, and error is around 1e-21,
                 # and when using the correct formula of energy efficiency it is around 1e+20
lambda_min = 0.000000001 
lambda_max = 1
max_iterations = 1000
sequential_gap = 1e-1 #0.00001
dichotomic_progression_ratio = 100 # progression based on  (max_lambda_candidate - min_labda_canditate) / dichotomic_progression_ratio
                                   # when choosing the next value of lambda.
                                   # this ratio is also used when exploring all possible values of lambda

#default values if not modified
number_of_iteration_during_exploration = 0
l_o_o_on_alpha_exploration = ltolerance
if search_ridge_coeff == False:  
    search_strategy = "----"
do_gaussian_process = True # ADDED before integrating experiment automatization
generate_plots = True  # MODIFIED before integrating experiment automatization
generate_best_configuration_file = False
number_of_best_combinations = 4000 # when generating the best configurations, this parameter should be used to limit the number of configurations to print in the output file 
process_database = False 
compute_marginal_effect = True   # MODIFIED before integrating experiment automatization
workstep = "computing_static_dynamic_score_for_paper" #"finding_best_input_dataset_size" #"computing_static_dynamic_score_for_paper" #"finding_best_input_dataset_size" 
           #"computing_static_dynamic_score_for_paper"# "plotting_graphs_for_the_paper"#"computing_static_dynamic_score_for_paper" 
             # "plotting_graphs_for_the_paper" # "processing_light_database" #"testing_best_configuration_estimation"#"increasing_samsung_data_set" #"testing_best_configuration_estimation" #"reviewing_mariginal_interaction_plots" # "reviewing_mariginal_interaction_plots" #"looking_strange_cases"
paper_fontsize = 35
acceptable_marginal_mean_value = 0 # if not 0, the horizontal bar at the level of this value will appear in marginal graph (on the samsung).
                                       # this can help decision maker to chose acceptable situation, before scheduling at thread on core 
                                       # or increasing socket frequency, associated with a parameter of the model. 
repeat_experiments = False
one_experiment = True

import time


if workstep == "looking_strange_cases" :
    print(" --- looking at strange cases")
    X_format_in_model = "base_Y"
    automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/can_be_reused/looking_at_strange_cases"
    value_to_retain = "mean" 
    remove_aberrant_points = False 
    do_gaussian_process = True #
    generate_plots = True  # 
    compute_marginal_effect = False   # 
elif search_ridge_coeff == False and phone_name == "google_pixel_4a_5g":
    automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/can_be_reused/GOOGLE_PIXEL_RESULTS/summary_files_only_0.89"
    if X_format_in_model == "base_Y":
        print(" --- reviewing marginal plot on google pixel")
        alpha = 0.01000000099
        energy_gap = 10 
        number_of_neighbour = 10   
        if workstep ==  "testing_best_configuration_estimation":
            do_gaussian_process = True #
            generate_plots = True  # 
            compute_marginal_effect = False   #
            generate_best_configuration_file = True
        if workstep == "processing_light_database":
            fill_data_from_folders = False
            process_database = True
            compute_marginal_effect = False   #
            generate_best_configuration_file = False
            r2_score_as_string = "0.89" # to write in the correct output file
        if workstep ==  "plotting_graphs_for_the_paper": 
            do_gaussian_process = True #
            generate_plots = True  # 
            compute_marginal_effect = True   #
            process_database = False  
        if workstep ==  "computing_static_dynamic_score_for_paper":
            fill_data_from_folders = True
            do_gaussian_process = True #
            generate_plots = True  # 
            process_database = True
            compute_marginal_effect = True   #
            generate_best_configuration_file = False
            r2_score_as_string = "0.89" # to write in the correct output file
        if workstep ==  "finding_best_input_dataset_size":
            one_experiment = False  
            repeat_experiments = True
            fill_data_from_folders = True
            do_gaussian_process = True #
            generate_plots = True  # 
            process_database = True
            compute_marginal_effect = True   #
            generate_best_configuration_file = False
            
            
    elif X_format_in_model == "base_Y_N_on_socket":
        alpha = 0.02000000098
        number_of_neighbour = 20
        time.sleep(5)
        automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/can_be_reused/GOOGLE_PIXEL_RESULTS/summary_files_only_0.89"
  

     
elif search_ridge_coeff == False and X_format_in_model == "base_Y" and phone_name == "samsung_galaxy_s8":

    print(" --- New tests on samsung galaxy s8")
    automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/can_be_reused/SAMSUNG_RESULTS/summary_files_only_samsung_before_considering_free_freq"
    # Note: the previous file is the one finally used for interpretations, the R2 score was 0.90
    #automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/can_be_reused/SAMSUNG_RESULTS/summary_files_only_samsung_0.92"
    alpha = 0.049940597311979425
    energy_gap = 10
    number_of_neighbour = 10
    acceptable_marginal_mean_value = -2.6e+9

    if workstep == "increasing_samsung_data_set" :
         automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/can_be_reused/SAMSUNG_RESULTS/summary_files_only_samsung_last_version"



elif search_ridge_coeff == False and X_format_in_model == "base_Y_F" and phone_name == "google_pixel_4a_5g":
    print(" --- Tests on google pixel including the ones with the frequency and the governor not modified")

elif search_ridge_coeff == False and X_format_in_model == "base_Y_F_N_on_socket" and phone_name == "google_pixel_4a_5g":
    print(" --- Tests on google pixel including the ones with the frequency and the governor not modified, we consider the number of threads on sockets")
    automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/can_be_reused/GOOGLE_PIXEL_RESULTS/all_summary_files_only__frequency_freed_and_not"



# parameter regarding outputs files
output_data_folder = "model_output_data/"
result_summary_csv_file_name = "best_R2_exploration_summary.csv"
marginal_effect_exploration_folder  = "marginal_effect_exploration_automatic_experiments" # Can change depending on the r2 score
linear_coeff_vs_kernel_ridge_margins_file = marginal_effect_exploration_folder + "/linear_coeff_vs_kernel_ridge_margins.csv" # Can change depending on the r2 score
start = time.time()
####################################  Prediction on google pixel 4a 5g



def function_to_remove_aberrant_points(X_user_friendly, X, y, output_data_folder, energy_array, workload_array, energy_gap, number_of_neighbour, repeat_experiments):
    print("--- Size of X before removing aberrants points from the dataset: ", len(X))
    X_len_before_removing_abberrant_points = len(X)
    utils.capture_X_y_in_file(np.array(X),  np.array(y), output_data_folder + "From_summaries_X_y_before_removing_aberrant_points.csv")
    print ("*** Total Configurations formatted 2: ", len(X))
    X_user_friendly, X, y = utils.remove_aberrant_points(X_user_friendly, X, y, energy_array, workload_array, energy_gap, number_of_neighbour, sigma_2 = (len(X[0]) + 1), repeat_experiments = repeat_experiments )
    utils.capture_X_y_in_file(np.array(X),  np.array(y), output_data_folder + "From_summaries_X_y_after_removing_aberrant_points.csv")
    print("--- Size of X after removing aberrants points from the dataset: ", len(X))
    print("--- Number of abberant points removed : ", X_len_before_removing_abberrant_points - len(X))
    print("*** Ratio energy by wokload : ", y)
    return X_user_friendly, X, y

def function_to_remove_duplicates(X_user_friendly, X, y, output_data_folder, energy_array, workload_array, value_to_retain):
    print("--- Size of X before removing duplicates: ", len(X))
    X_len_before_removing_duplicates = len(X)
    utils.capture_X_y_in_file(np.array(X),  np.array(y), output_data_folder + "From_summaries_X_y_before_removing_duplicate.csv")
    X_user_friendly, X, y = utils.remove_duplicates(X_user_friendly, X, y, energy_array, workload_array, value_to_retain)
    utils.capture_X_y_in_file(np.array(X),  np.array(y), output_data_folder + "From_summaries_X_y_after_removing_duplicate.csv")
    print("--- Size of X after removing duplicates: ", len(X))
    print("--- Number of duplicates points removed : ", X_len_before_removing_duplicates - len(X))
    print("*** Ratio energy by wokload : ", y)
    return X_user_friendly, X, y


                           
                        
                
def function_to_fill_data_from_folders(consider_automatization_summaries,
                automatization_summaries_folder, max_input_size, energy_gap, number_of_neighbour,
                 input_format, consider_exact_values_of_frequency, X_format_in_model, 
                 phone_name):

    if consider_automatization_summaries:
        X_user_friendly = utils.get_data_from_summary_folder(automatization_summaries_folder, "configurations", "human_readable_format", maximum_input_size = max_input_size )
        print ("*** Total configurations in user friendly format: ", X_user_friendly)
        X = utils.get_data_from_summary_folder(automatization_summaries_folder, "configurations", input_format, consider_exact_values_of_frequency, X_format_in_model, maximum_input_size = max_input_size)
        print ("*** Total Configurations formatted: ", X) # From this point, X in in Base Y format.
        X_dict = utils.create_dictionnary_for_X(X_user_friendly, X)
        print ("*** Total Configurations dictionnary: ", X_dict)
        y = utils.get_data_from_summary_folder(automatization_summaries_folder, "energy_efficiency", maximum_input_size = max_input_size )
        print ("*** Total energy efficiencies: ", y)
        energy_array =  utils.get_data_from_summary_folder(automatization_summaries_folder, "energy", maximum_input_size = max_input_size )
        print ("*** Total energy : ", energy_array)
        print("*** Sum of energy:", sum(energy_array))
        workload_array =  utils.get_data_from_summary_folder(automatization_summaries_folder, "workload", maximum_input_size = max_input_size )
        print ("*** Total workload : ", workload_array)
    else:
        import fill_data_from_manual_experiments as data
        X_user_friendly = data.get_X(phone_name, "human_readable_format")
        print ("*** Total configurations in user friendly format: ", X_user_friendly)
        X = data.get_X(phone_name, input_format, consider_exact_values_of_frequency)
        print ("*** Total Configurations formatted: ", X)
        X_dict = data.get_X_dict(phone_name, input_format, consider_exact_values_of_frequency)
        print ("*** Total Configurations dictionnary: ", X_dict)
        y = data.get_y(phone_name)
        print ("*** Total energy efficiencies: ", y)

    if remove_aberrant_points : 
       X_user_friendly, X, y = function_to_remove_aberrant_points(X_user_friendly, X, y, output_data_folder,  energy_array, workload_array, 
                         energy_gap, number_of_neighbour, repeat_experiments)
    if remove_duplicates:
       X_user_friendly, X, y = function_to_remove_duplicates(X_user_friendly, X, y, output_data_folder, energy_array, workload_array, value_to_retain)

    return X_user_friendly, X, X_dict, y, energy_array, workload_array


def function_to_train_the_model(phone_name, energy_gap, number_of_neighbour, dataset_size_to_consider,  X_user_friendly, X, X_dict, y, energy_array, workload_array,
                 input_format, consider_exact_values_of_frequency, X_format_in_model, 
                  populate_inputs_to_considere_thread_combinations_on_same_socket, 
                 maximum_number_of_combination, one_hot_encoding_of_frequency,standartize_inputs,
                 search_ridge_coeff, search_strategy, number_of_iteration_during_exploration, l_o_o_on_alpha_exploration, ltolerance,lambda_min, max_iterations,sequential_gap,
                 dichotomic_progression_ratio, 
                 generate_plots, result_summary_csv_file_name, alpha, 
                 repeat_experiments):

    ratio_to_throw_away = (len(X) - dataset_size_to_consider)/len(X)
    if ratio_to_throw_away > 0:
        X_to_considered, X_to_thrown, y_to_considered, y_to_thrown = train_test_split(X, y, test_size = ratio_to_throw_away, random_state=2)
        print ("Set to consider (size = " + str(len(X_to_considered) )+  ") : ", X_to_considered)
        print ("energy by workload to consider  (size = " + str(len(y_to_considered) )+  ") : ", y_to_considered)
        print ("Set to thrown (size = " + str(len(X_to_thrown) )+  ") : ", X_to_thrown)
        print ("energy by workload to thrown  (size = " + str(len(y_to_thrown) )+  ") : ", y_to_thrown)
        X = X_to_considered
        y = y_to_considered

    print("--- Total energy consumed :", energy_array)
    #################################

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=2)
    X_user_friendly_train = utils.get_X_user_friendly_from_X(X_train, X_dict)
    X_user_friendly_test = utils.get_X_user_friendly_from_X(X_test, X_dict)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    y_train_before_populating = y_train
    X_train_before_populating = X_train

    print ("Train set Configurations : ", X_train)
    print ("Train set energy by workload : ", y_train)
    print ("Test set Configurations : ", X_test)
    print ("Test set energy by workload : ", y_test)
    print ("Train set Configurations in user friendly mode : ", X_user_friendly_train)
    print ("Test set Configurations in user friendly mode : ", X_user_friendly_test)


    print ("Size of X_train: ", len(X_train))
    print ("Size of X_test: ", len(X_test))

    # getting parameter sigma_2
    sigma_2 = len(X_train[0]) 

    # added for populating the input datas, considering that the outcom is the same independantly of the combination of threads on the same socket. 
    if(populate_inputs_to_considere_thread_combinations_on_same_socket):
        X_train_populated, y_train_populated = utils.populate_input_datas(X_train, y_train, input_format, maximum_number_of_combination)
        X_train = np.asarray(X_train_populated)
        y_train = np.asarray(y_train_populated)
        utils.capture_X_y_in_file(X_train,  y_train, output_data_folder + "X_y_populated" + str(maximum_number_of_combination) + ".csv")
        
    # if one hot encoding of frequency
    if (one_hot_encoding_of_frequency):
        X_train = utils.sparce_X_to_get_impact_of_frequency(X_train)  
        X_test = utils.sparce_X_to_get_impact_of_frequency(X_test)  

    if(standartize_inputs):
        print (" ---> Standartization")
        X_train_ = utils.standartize(X_train)
        print(" Original X", X_train)
        print ("Standartized X", X_train_)
        X_train = X_train_
        print("---> end of standartization")

    # search ridge coeff
    if(search_ridge_coeff):
        print("---> lambda exploration")
        alpha, number_of_iteration_during_exploration, l_o_o_on_alpha_exploration =  utils.find_regularization_parameter(X_train, y_train, 
                                                        sigma_2, search_strategy, ltolerance , lambda_min , lambda_max , max_iterations , sequential_gap , dichotomic_progression_ratio)
        print("---> end added for lambda exploration") 
        print ("Train set, energy by workload : ", y_train)

    if (do_gaussian_process):
        ############## now using kernel ridge to train data
        #A good point is that this implementation is avalaible online (see deeply below)  and from the source code refractoring we can obtain 
        # the ci values which is the attribute "parest". We can also have sigma2, which is the "scale", default value is 0.5 . 
        # NOTE: use np.asarray(X/Y) on your inputs if you have this ERROR TypeError: list indices must be integers or slices, not tuple
        # The default value of the ridge coefficient can be 1 because I tooked this value from the sklearn source code implementation, they called it alpha https://github.com/scikit-learn/scikit-learn/blob/229bd226ab0d18c0ea84c598cd87d86f599eaac8/sklearn/kernel_ridge.py
        # 
        print (" *****  Training the datas ***** ")
        # start the code of gaussian process to uncomment

        gauss_process = GaussProcess(X_train, y_train,
                            scale = sigma_2, ridgecoeff = alpha)
        gauss_process.fit(y_train)
        predicted_y_test = gauss_process.predict(np.asarray(X_test))
        print(" **** Predicted y test = ", predicted_y_test)
        print(" Kernel ridge R2 score = ", utils.compute_r2_score(y_test, predicted_y_test))


         # generating linear regression coefficients
        ols = sm.OLS(y_train, X_train)  # warning in the sm OLS function argument format, y is the first parameter. 
        reg = ols.fit()
        reg_pred_y_test = reg.predict(X_test)
        linear_coefficients_ = reg.params
        print("Predicted y test = ", reg_pred_y_test)
        print("linear model parameters  = ",  linear_coefficients_)
        print("*** Linear model R2 score  = ", utils.compute_r2_score(y_test, reg_pred_y_test) )
        linear_R2_score =  utils.compute_r2_score(y_test, reg_pred_y_test)
    
    # printing plots
    if (generate_plots): 
        print ("printing plots")
        _, (orig_data_ax, testin_data_ax, kernel_ridge_ax) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 13))
        orig_data_ax.bar(X_user_friendly_train,y_train_before_populating, width=0.4)
        # Add title and axis names

        """
        kernel_ridge_ax.bar(utils.get_X_user_friendly_from_X(X, X_dict),y, width=0.4)


        kernel_ridge_ax.set_title('All energy/workload ratio')
        kernel_ridge_ax.set_xlabel('Configuration')
        kernel_ridge_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
        kernel_ridge_ax.tick_params(size=8)
        _ = kernel_ridge_ax.set_title("All datas")
        """
        orig_data_ax.set_title('Training datas energy/workload ratio')
        orig_data_ax.set_xlabel('Configuration')
        orig_data_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
        orig_data_ax.tick_params(size=8)

        testin_data_ax.bar(X_user_friendly_test,y_test, width=0.4)
        # Add title and axis names
        testin_data_ax.set_title('Testing datas energy/workload ratio')
        testin_data_ax.set_xlabel('Configuration')
        testin_data_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
        testin_data_ax.tick_params(size=8)


        kernel_ridge_ax.bar(X_user_friendly_test,predicted_y_test, width=0.4)
        # Add title and axis names
        kernel_ridge_ax.set_title('Predited energy/workload ratio')
        kernel_ridge_ax.set_xlabel('Configuration')
        kernel_ridge_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
        kernel_ridge_ax.tick_params(size=8)

        if workstep == "looking_strange_cases":
            # Add title and axis names
            print ("Size of X ---: ", len(X))
            print ("Size of y ---: ", len(y))
            print ("X ---: ", X)
            print ("y ---: ", y)
            kernel_ridge_ax.bar(utils.get_X_user_friendly_from_X(X, X_dict),y, width=0.4)

            kernel_ridge_ax.set_title('All energy/workload ratio')
            kernel_ridge_ax.set_xlabel('Configuration')
            kernel_ridge_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
            kernel_ridge_ax.tick_params(size=8)
            _ = kernel_ridge_ax.set_title("All datas")

        else : 
            
            kernel_ridge_ax.bar(X_user_friendly_test,predicted_y_test, width=0.4)
            # Add title and axis names
            kernel_ridge_ax.set_title('Predited energy/workload ratio')
            kernel_ridge_ax.set_xlabel('Configuration')
            kernel_ridge_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
            kernel_ridge_ax.tick_params(size=8)
            _ = kernel_ridge_ax.set_title("Predicted data\n using kernel ridge, R2 = " + str(utils.compute_r2_score(y_test, predicted_y_test)))

        R2_score = utils.compute_r2_score(y_test, predicted_y_test)
        print(" R2 error = ", utils.compute_r2_score(y_test, predicted_y_test))
        if(search_ridge_coeff):
            print("kernel ridge coef (lambda) = ", alpha)
            print("number of iteration on kernel ridge coef = ", number_of_iteration_during_exploration) 
            print("leave_one_out error on kernel ridge coef = ", l_o_o_on_alpha_exploration)
        

        plt.gcf().autofmt_xdate()
        plt.xticks(fontsize=8)
        plt.savefig("kernel_ridge_prediction_on_google_pixel_4a_5g.png")
        plt.clf()
        plt.cla()
        plt.close()
        #end of the code of gaussian process to uncomment
        experiments_preprocessing_parameters = [phone_name, 
                            input_format,
                        consider_exact_values_of_frequency ,
                            populate_inputs_to_considere_thread_combinations_on_same_socket,
                            maximum_number_of_combination,
                            alpha,
                        search_ridge_coeff,
                        search_strategy,
                        number_of_iteration_during_exploration,
                        l_o_o_on_alpha_exploration,
                        ltolerance,
                        lambda_min,
                        max_iterations,
                        sequential_gap,
                        dichotomic_progression_ratio,
                        standartize_inputs,
                        R2_score, energy_gap, number_of_neighbour, X_format_in_model 
        ] 
        utils.write_result_in_csv_file(result_summary_csv_file_name, experiments_preprocessing_parameters)
        # folder preparation, for evantual marginal effect or other explorations
        r2_score_as_string = repr(R2_score)
        if repeat_experiments:  # for the paper we are looking for the best input dataset size
            marginal_effect_exploration_folder  = "finding_best_input_dataset_size/marginal_effect_exploration_" +\
                             phone_name [0:7] + "_" + r2_score_as_string [0:4] + "_" + X_format_in_model  # Can change depending on the r2 score
        else :
            marginal_effect_exploration_folder  = "marginal_effect_exploration_automatic_experiments_" +\
                            phone_name [0:7] + "_" + r2_score_as_string [0:4] + "_" + X_format_in_model  # Can change depending on the r2 score

        os.makedirs(marginal_effect_exploration_folder, exist_ok=True)

        if generate_best_configuration_file:
            utils_for_validation.get_and_write_sorted_estimations_of_configurations(number_of_best_combinations, 
                        gauss_process, input_format, X_format_in_model, consider_exact_values_of_frequency,
                output_file_path = marginal_effect_exploration_folder + "/energy_efficiency_estimations.csv")


    return gauss_process,  X_user_friendly, X, y, energy_array, workload_array, X_train, y_train, X_test, y_test, sigma_2, marginal_effect_exploration_folder, R2_score, linear_R2_score

def function_to_compute_marginal_effect(gauss_process, X_train, y_train, X_test, y_test, sigma_2, marginal_effect_exploration_folder, R2_score, 
              base_Y__X_meaning_dictionnary, base_Y_N_on_socket__X_meaning_dictionnary, base_Y_F__X_meaning_dictionnary, base_Y_F_N_on_socket__X_meaning_dictionnary,
            X_format_in_model, workstep, phone_name, repeat_experiments = False): 

    # computing marginal effect based on formulas (7) and (8) of the paper Kernel-Based Regularized Least Squares in R (KRLS) and Stata (krls). 

    # Note : The index i represent the observation and j represent the variable. 
    #        we have N observations and M variables
    # getting the coef_i vector and sigma_2, in this case, I don't know why the format returned by gauss_process.parest is not the same as in this exemple,
    #              so I add M lines, to have the format (M,1), where M is the number of observation.  
    c_vector = (gauss_process.parest)[:,np.newaxis]
    print("Computed c values  = ", c_vector)

    # computing the marginal effect of the observation with "optimized" approaoch
    print (" ***** START computing marginal effects with matrix***** ")
    print ("X = ", X_train)
    pointwise_margins, margins = comput_margin.marginal_effect(X_train, c_vector, sigma_2, repeat_experiments)
    print (" ***** END computing marginal effects ***** ")
    print("margins",  margins)
    print("pointwise margins",  pointwise_margins)

    # generating linear regression coefficients
    ols = sm.OLS(y_train, X_train)  # warning in the sm OLS function argument format, y is the first parameter. 
    reg = ols.fit()
    reg_pred_y_test = reg.predict(X_test)
    linear_coefficients = reg.params
    print("Predicted y test = ", reg_pred_y_test)
    print("linear model parameters  = ",  linear_coefficients)
    print("*** Linear model R2 score  = ", utils.compute_r2_score(y_test, reg_pred_y_test) )
    #linear_R2_score =  utils.compute_r2_score(y_test, reg_pred_y_test)
      
    if phone_name == "google_pixel_4a_5g" :
        linear_coeff_vs_kernel_ridge_margins_file = marginal_effect_exploration_folder + "/linear_coeff_vs_kernel_ridge_margins.csv" # Can change depending on the r2 score
        if X_format_in_model == "base_Y":
            if( workstep == "plotting_graphs_for_the_paper"):
                X_meaning_dictionnary = utils.get_for_the_paper_X_format_meaning_dictionnaries(phone_name)
            else: 
                X_meaning_dictionnary = base_Y__X_meaning_dictionnary
            utils.capture_kernel_means_marginal_and_linear_model_coeff(margins, linear_coefficients, linear_coeff_vs_kernel_ridge_margins_file, X_meaning_dictionnary)
        
            
            ### Plotting X_1 distribution plot (Note, it is the activation state of the first core! because we are in Base_Y format of X).
            # plotting histograph
        
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,0], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_little_socket_frequency_level.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,1], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_0_state.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,7], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_6_frequency_level.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,8], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_7_frequency_level.png", 3e-11, 8)

           
            

            ## Regression of d_X_1 over all other variable including 
            print(" X train size: ", len(X_train))
            print(" Margin size ", len(pointwise_margins[:,1]))
            print(" Repeat experiments ", repeat_experiments)
            d_X_1_coefficients_file = marginal_effect_exploration_folder + "/d_X_1_linear_coefficients.csv"
            d_X_1_ols = sm.OLS(pointwise_margins[:,1], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_1_reg = d_X_1_ols.fit()
            d_X_1_linear_coefficients = d_X_1_reg.params
            print("X_0_d linear model parameters  = ",  d_X_1_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 1, 
                        d_X_i_linear_coefficients = d_X_1_linear_coefficients, 
                            file_path = d_X_1_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_0 over other variables")
            avg_marginal_score_table = utils.plot_marginal_interactions(X_train, pointwise_margins, 1, 0, 1, 2, 3,4,5,6,7, 8, X_meaning_dictionnary, marginal_effect_exploration_folder, 
                                 workstep = "plotting_graphs_for_the_paper", paper_fontsize = 28)
            
    
    return pointwise_margins, margins, X_meaning_dictionnary      
  



def column(matrix, i):
    result = []
    for row in matrix: 
        print(" --- First column of the row ", row[i])
        result.append(row[i])
    return result
def special_sum(my_list):
    # This function is to avoid None values entry when computing the sum of an array 
    result = 0
    for i in my_list: 
        print(" --- Computing special sum and adding value ", i)
        if i is not None:
            result = result + i
    return result

def special_len(my_list):
    # This function is to avoid None values entry when computing the sum of an array 
    result = 0
    for i in my_list: 
        print(" --- Computing special sum and adding value ", i)
        if i is not None:
            result = result + 1
        else: 
            print("---- " + str(i) + " is not added")
    return result




def compute_global_lin_reg_coef_and_abs_coef_table(X_train, pointwise_margins):
    # return for all j an  array contaning at each position the list
    #  [l1_coef,L2_coef,  ..., LM coef]
    #  [ L1_abs_coef, L2_abs_coef, ... LM_abs_coef]
    #  
    #      
    #   
    gobal_lin_reg_coef_table = []
    for i in range(0,9): 
        d_y_on_d_X_i_ols = sm.OLS(pointwise_margins[:,1], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
        d_y_on_d_X_i_reg = d_y_on_d_X_i_ols.fit()
        d_y_on_d_X_i_linear_coefficients = d_y_on_d_X_i_reg.params

        coef_array = [float(element) for element in d_y_on_d_X_i_linear_coefficients]
        abs_coef_array = [abs(float(element)) for element in d_y_on_d_X_i_linear_coefficients]

        gobal_lin_reg_coef_table.append([coef_array, abs_coef_array])
    return gobal_lin_reg_coef_table




def compute_global_avg_dynamic_score_table(conn, X_train, pointwise_margins, X_meaning_dictionnary, marginal_effect_exploration_folder, repeat_experiments):
    global_avg_margins_and_dynamic_score_table = []
    for i in range(0,9): 
        #for i in [1]:                              
        avg_margin_table = utils.plot_marginal_interactions(X_train, pointwise_margins, i, 0, 1, 2, 3,4,5,6,7, 8, X_meaning_dictionnary, marginal_effect_exploration_folder,
                     workstep = "computing_static_dynamic_score_for_paper", paper_fontsize = 28, repeat_experiments = repeat_experiments)
        print("--- Avg margin table " + str(i) + ": ", avg_margin_table ) 
        avg_margin_and_dynamic_score_table = utils_for_validation.validate_lesson_learned(conn, marginal_effect_exploration_folder, 
                    avg_marginal_score_table = avg_margin_table)      
        global_avg_margins_and_dynamic_score_table.append(avg_margin_and_dynamic_score_table)
        print("--- Interaction table and validation scores of variable " + str(i) + ": ", avg_margin_and_dynamic_score_table )
    return global_avg_margins_and_dynamic_score_table

def get_color_map(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

import sys
def plot_finalscore_as_a_function_of_best_margin_interact(global_avg_margins_and_dynamic_score_table,
                                     gobal_lin_reg_coef_table, margins, dataset_size):
    output_plot_name = "dynamic_score_scheduling_size_according_to_best_covariates_and_best_interactions_" + str(dataset_size) +".png"
    numbers_of_selected_margins = np.arange(start=1, stop=10, step=1).tolist() # We just limit our self to all the margins
    numbers_of_selected_interactions = np.arange(start=1, stop=10, step=1).tolist()
    
    score_data_set_to_plot = []
    size_data_set_to_plot = []
    x_of_data_set = []
    for i in numbers_of_selected_margins:
        one_line_size_data_set = []
        one_line_score_data_set = []
        one_x_line_of_data_set = []
        for j in numbers_of_selected_interactions:
            final_score, selected_global_table , table_size = compute_final_score_for_strong_interaction_and_margins(global_avg_margins_and_dynamic_score_table,
                                     gobal_lin_reg_coef_table, margins, i, j, dataset_size)
                                
            if final_score is not None :
                one_line_score_data_set.append(final_score)
                #one_line_size_data_set.append(sys.getsizeof(selected_global_table))
                one_line_size_data_set.append(table_size/1000)
                one_x_line_of_data_set.append(j)
        if(len(one_line_size_data_set) > 0 ):
            size_data_set_to_plot.append(one_line_size_data_set)
            score_data_set_to_plot.append(one_line_score_data_set)
            x_of_data_set.append(one_x_line_of_data_set)
    paper_fontsize = 20

    
    fig, ((scores_according_to_margins, score_and_size) ) = plt.subplots(nrows= 1, ncols = 2, sharex = True, figsize=(15, 7))
    color_map = get_color_map(len(score_data_set_to_plot))
    for i in range(0,len(score_data_set_to_plot)):
        scores_according_to_margins.plot(x_of_data_set[i], score_data_set_to_plot[i] , label = r'$m_J$ = ' + str(i+1) , color= color_map(i),marker='o', linestyle='-')
    scores_according_to_margins.set_xlabel("Number of secondary covariates L \n for J" + r'$\leftrightarrow$ ' +"L interactions",  fontsize = paper_fontsize )
    scores_according_to_margins.set_ylabel( "FitsAll dynamic score", fontsize = paper_fontsize)
    scores_according_to_margins.tick_params(axis='x', which='major' , labelsize= paper_fontsize)
    scores_according_to_margins.tick_params(axis='y', which='major' , labelsize= paper_fontsize)
    scores_according_to_margins.tick_params(size=8)
    scores_according_to_margins.legend(loc = 'upper left', prop={'size': 18})
    scores_according_to_margins.set_title("Dynamic score computed with a different \n number " +  r'$m_J$'+ " of first covariates J selected", fontsize = paper_fontsize)
        
   
    
    choosen_number_of_margins = 7-1  # -1 to get the index in dataset
    score_and_size.plot(x_of_data_set[choosen_number_of_margins], score_data_set_to_plot[choosen_number_of_margins] ,  color='black',marker='o', linestyle='-' )
    #ax.set_xlabel( "For each covariate, number of other \n  covariates considered when selecting interactions." ,  fontsize = paper_fontsize )
    score_and_size.set_ylabel( "FitsAll dynamic score", fontsize = paper_fontsize)
    score_and_size.tick_params(axis='x', which='major' , labelsize= paper_fontsize)
    score_and_size.tick_params(axis='y', which='major' , labelsize= paper_fontsize)
    score_and_size.set_xlabel("Number of secondary covariates L \n for J" + r'$\leftrightarrow$ ' +"L interactions", fontsize = paper_fontsize)

    score_and_size.tick_params(size=8)
    score_and_size.set_title( "Dynamic score (in black) and \n scheduling data base size (in blue)", fontsize = paper_fontsize)

    ax2=score_and_size.twinx()
    ax2.plot(x_of_data_set[choosen_number_of_margins], size_data_set_to_plot[choosen_number_of_margins] ,  color="blue", marker='o', linestyle='-' )
    ax2.set_xlabel("Number of secondary covariates L \n for J" + r'$\leftrightarrow$' +"L interactions", color="blue", fontsize = paper_fontsize)
    ax2.set_ylabel("Scheduling database size (KBytes)", color="blue", fontsize = paper_fontsize)

    #ax2.tick_params(axis='x', which='major' , labelsize= paper_fontsize, color="gray",)
    ax2.tick_params(axis='y', which='major' , labelsize= paper_fontsize, color="Blue")
    ax2.tick_params(size=8)

    # Get extents of subplot
    
    #plt.subplots_adjust(bottom = 0.15)
    #fig.text(0.5, 0.04, "For each covariate J, number of other  covariates considered when selecting interactions J " + r'$\leftrightarrow$ = ' +" L.", 
    #       va='center', ha='center', fontsize=paper_fontsize)
    #plt.yticks(fontsize=paper_fontsize)
    #plt.legend(loc="upper left",  fontsize = 12)
    #ax = plt.gca()
    #ax.yaxis.get_offset_text().set_fontsize(paper_fontsize)
    #plt.locator_params(axis='x', nbins=4)
    plt.tight_layout(pad = 1.5)  
    #output_plot_name = "dynamic_score_according_to_best_covariates_and_best_interactions" +str(j)+ ".png"     
    plt.savefig("finding_best_input_dataset_size/"+ output_plot_name )
    plt.clf()
    plt.cla()
    plt.close()
    


def compute_final_score_for_strong_interaction_and_margins(global_avg_margins_and_dynamic_score_table, gobal_lin_reg_coef_table, 
                 margins,  n_best_covariates = 9, n_strong_interact = 9, dataset_size = 536):
    final_score = 0
    # This function first sort each single table in the linear coef global table according to the coefficient of secondary covariates. 
    # secondly it sort the covariate lists occording to thier means margins
    #  it compute the dynamic score on the dynamic global table, by retaining only best secondary covariates. 
    # The best secondly are related to previous computed table 
    # step 1: sorting linear coefs
    sorted_gobal_lin_reg_coef_table = []
    selected_global_avg_margins_and_dynamic_score_table = []
    table_size = 0
    for j in range(0,9):                                                                        #1 because we take the abs coeff values table
        sorted_table_second_cov_rank__coef_value =  sorted(enumerate(gobal_lin_reg_coef_table[j][1]), key=lambda kv: kv[1],  reverse=True) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
        sorted_gobal_lin_reg_coef_table.append(sorted_table_second_cov_rank__coef_value)
    
    # step 2: sorting margins
    sorted_means_margins = []
    sorted_means_margins =  sorted(enumerate(margins), key=lambda kv: kv[1],  reverse=True) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
    

    #step 3: scomputing the score 
    dynamic_score_list = []
    number_of_dynamic_scores = 0
    total_of_dynamic_scores = 0
    considered_margin_couples = sorted_means_margins[0:n_best_covariates]
    considered_margins_index = column(considered_margin_couples,0)  # column of index of selected (Lindice, L_coef) couples
    print("--- considered margins couples ", considered_margin_couples)
    for j in considered_margins_index: 
        considered_l_couples = sorted_gobal_lin_reg_coef_table[j][0:n_strong_interact]
        considered_l_index = column(considered_l_couples,0)  # column of index of selected (Lindice, L_coef) couples
        print("--- considered L (index, coef) couples ", considered_l_couples)
        selected_avg_margins_and_dynamic_score_table = []  # to compute the selected table
        for L_entry_index in considered_l_index:
            L_entry = global_avg_margins_and_dynamic_score_table[j][L_entry_index]  # now we have an L_entry, with l values, l_avg values and dynamic scores 
            score_and_transitions_list = L_entry[2]
            print("--- Only scores and transitions, just to verify : ", score_and_transitions_list) 
            dynamic_score_list.append(column(score_and_transitions_list,0))
            number_of_dynamic_scores = number_of_dynamic_scores + special_len(column(score_and_transitions_list,0))  # columns because,  we test only the score, if it is None
            total_of_dynamic_scores = total_of_dynamic_scores + special_sum(column(score_and_transitions_list,0))  # because score_and_transitions_list is a matrice N_Lvalue * 3  (3 for score , positive_transition, negative_transition)
            selected_avg_margins_and_dynamic_score_table.append([L_entry_index, L_entry]) # we add the index of the variable L to hepl scheduler to repair it
            table_size = table_size + sys.getsizeof(L_entry_index)  + sys.getsizeof(L_entry[0]) + sys.getsizeof(L_entry[1]) # L values and interactions avg 
        selected_global_avg_margins_and_dynamic_score_table.append([j, selected_avg_margins_and_dynamic_score_table]) # Note, the final global table does not have the same format than the original one
                                                                                 # Because the means margins have been ranked, the J index should be also saved, so each 
                                                                                 # entry in the final global table should have [j_index, corresponding list of L_entries]
                                                                                 # it is the same principle applyed with the L_entry_index above.
    if(number_of_dynamic_scores == 0):
        return None, None, None
    final_score =  (total_of_dynamic_scores/number_of_dynamic_scores)/100 
    print("--- AVG and Scores of all selected interactions", global_avg_margins_and_dynamic_score_table) 
    print("--- Dynamic scores only : ", dynamic_score_list )
    print("--- Dtatset_size: ", dataset_size)
    print("--- N Best covariates = " + str(n_best_covariates) + ", N Best interactions = " + str(n_strong_interact))
    print("--- Final selected table: ", selected_global_avg_margins_and_dynamic_score_table)
    print("--- size in scheduling database ", table_size)
    print("--- Final dynamic score : ", final_score)
    return final_score, selected_global_avg_margins_and_dynamic_score_table, table_size


def funtion_to_process_database( marginal_effect_exploration_folder , X_user_friendly, X, y, energy_array, workload_array, 
  X_train, y_train, X_test, y_test, gauss_process, pointwise_margins, margins, X_meaning_dictionnary, dataset_size = 536, repeat_experiments = False, 
  n_best_covariates = 7, n_strong_interact = 7, acceptance_degree = 170):
    # this function write and make valilation on the database
    os.makedirs(marginal_effect_exploration_folder, exist_ok=True)
    lesson_learned_validation_output_file = "lesson_learned_validation_file.csv"
    static_score_plot_file = "static_score_plot.png"

    print("--- Creating / opening data base")
    conn = sqlite3.connect('experiments_and_estimations_results.db')

    if repeat_experiments : 
        conn.close()
        print("--- Creating / opening data base")
        conn = sqlite3.connect(':memory:')   # ':memory:' is a keyword to create an in memory database.
        print("--- Opened database successfully")
        utils_for_validation.create_database_and_tables(conn)
        utils_for_validation.fill_database(conn, X_user_friendly, X, y, energy_array, workload_array,  X_train, y_train, X_test, y_test, gauss_process, table_name = "all")

    global_avg_margins_and_dynamic_score_table = compute_global_avg_dynamic_score_table(conn,  X_train, pointwise_margins, X_meaning_dictionnary, 
                     marginal_effect_exploration_folder, repeat_experiments)
    gobal_lin_reg_coef_table = compute_global_lin_reg_coef_and_abs_coef_table(X_train, pointwise_margins)
   
    # to uncomment
    plot_finalscore_as_a_function_of_best_margin_interact(global_avg_margins_and_dynamic_score_table,
                                    gobal_lin_reg_coef_table, margins, dataset_size)

    final_score, selected_global_table, table_size = compute_final_score_for_strong_interaction_and_margins(global_avg_margins_and_dynamic_score_table, 
                          gobal_lin_reg_coef_table, margins, n_best_covariates, n_strong_interact, dataset_size)

    static_score =  utils_for_validation.compute_static_score (conn, "finding_best_input_dataset_size",  paper_fontsize = 20, 
                                     dataset_size = dataset_size, acceptance_degree = acceptance_degree)   
                                                             
    conn.close()
    return final_score, static_score



def get_dynamic_and_R2_score( phone_name, energy_gap, number_of_neighbour,
                dataset_size_to_consider,  X_user_friendly, X, X_dict, y, energy_array, workload_array,
                 input_format, consider_exact_values_of_frequency, X_format_in_model, 
                 populate_inputs_to_considere_thread_combinations_on_same_socket, 
                 maximum_number_of_combination, one_hot_encoding_of_frequency,standartize_inputs,
                 search_ridge_coeff, search_strategy, number_of_iteration_during_exploration, l_o_o_on_alpha_exploration, ltolerance,lambda_min, max_iterations,sequential_gap,
                 dichotomic_progression_ratio, 
                 generate_plots, result_summary_csv_file_name, alpha, 
                 repeat_experiments = False , n_best_covariates = 7, n_strong_interact =7, acceptance_degree = 170):
   

    gauss_process,  X_user_friendly, X, y, energy_array, workload_array, X_train, y_train, X_test, y_test, sigma_2, marginal_effect_exploration_folder, R2_score , linear_R2_score \
                = function_to_train_the_model(phone_name, energy_gap, number_of_neighbour, 
                dataset_size_to_consider,  X_user_friendly, X, X_dict, y, energy_array, workload_array,
                 input_format, consider_exact_values_of_frequency, X_format_in_model, 
                  populate_inputs_to_considere_thread_combinations_on_same_socket, 
                 maximum_number_of_combination, one_hot_encoding_of_frequency,standartize_inputs,
                 search_ridge_coeff, search_strategy, number_of_iteration_during_exploration, l_o_o_on_alpha_exploration, ltolerance,lambda_min, max_iterations,sequential_gap,
                 dichotomic_progression_ratio, 
                 generate_plots, result_summary_csv_file_name, alpha, 
                 repeat_experiments)
    
    pointwise_margins, margins, X_meaning_dictionnary = function_to_compute_marginal_effect(gauss_process, X_train, y_train, X_test, y_test, sigma_2, marginal_effect_exploration_folder, R2_score, 
              base_Y__X_meaning_dictionnary, base_Y_N_on_socket__X_meaning_dictionnary, base_Y_F__X_meaning_dictionnary, base_Y_F_N_on_socket__X_meaning_dictionnary, 
            X_format_in_model, workstep, phone_name, repeat_experiments)  
        
    dynamic_score, static_score = funtion_to_process_database( marginal_effect_exploration_folder , X_user_friendly, X, y, energy_array, workload_array,
       X_train, y_train, X_test, y_test, gauss_process, pointwise_margins, margins, X_meaning_dictionnary, dataset_size_to_consider, repeat_experiments, 
        n_best_covariates, n_strong_interact, acceptance_degree = acceptance_degree)
    return  R2_score, dynamic_score, linear_R2_score, static_score



if one_experiment: 
    if fill_data_from_folders:
        X_user_friendly, X, X_dict, y, energy_array, workload_array =  function_to_fill_data_from_folders(consider_automatization_summaries,
                automatization_summaries_folder, max_input_size, energy_gap, number_of_neighbour,
                 input_format, consider_exact_values_of_frequency, X_format_in_model, 
                 phone_name)

        gauss_process,  X_user_friendly, X, y, energy_array, workload_array, X_train, y_train, X_test, y_test, sigma_2, marginal_effect_exploration_folder, R2_score, linear_R2_score \
                = function_to_train_the_model(phone_name,  energy_gap, number_of_neighbour, len(X),  X_user_friendly, X, X_dict, y, energy_array, workload_array,
                 input_format, consider_exact_values_of_frequency, X_format_in_model, 
                 populate_inputs_to_considere_thread_combinations_on_same_socket, 
                 maximum_number_of_combination, one_hot_encoding_of_frequency,standartize_inputs,
                 search_ridge_coeff, search_strategy, number_of_iteration_during_exploration, l_o_o_on_alpha_exploration, ltolerance,lambda_min, max_iterations,sequential_gap,
                 dichotomic_progression_ratio, 
                 generate_plots, result_summary_csv_file_name, alpha, 
                 repeat_experiments)
       
    if compute_marginal_effect:   
        pointwise_margins, margins, X_meaning_dictionnary = function_to_compute_marginal_effect(gauss_process, X_train, y_train, X_test, y_test, sigma_2,
                marginal_effect_exploration_folder, R2_score, 
                base_Y__X_meaning_dictionnary, base_Y_N_on_socket__X_meaning_dictionnary, base_Y_F__X_meaning_dictionnary, base_Y_F_N_on_socket__X_meaning_dictionnary, 
                X_format_in_model, workstep, phone_name, repeat_experiments)  
                        
    # parameter are :
    if process_database:
        dynamic_score = funtion_to_process_database( marginal_effect_exploration_folder , X_user_friendly, X, y, energy_array, workload_array,  
                  X_train, y_train, X_test, y_test, gauss_process, pointwise_margins, margins,  X_meaning_dictionnary, len(X_train) + len(X_test), repeat_experiments)

import math
if repeat_experiments:

    X_user_friendly, X, X_dict, y, energy_array, workload_array =  function_to_fill_data_from_folders(consider_automatization_summaries,
                automatization_summaries_folder, max_input_size, energy_gap, number_of_neighbour,
                 input_format, consider_exact_values_of_frequency, X_format_in_model, 
                 phone_name)
    n_best_covariates = 7
    n_strong_interact = 3  
    acceptance_ratio = 0.317
    # 170          
    output_plot_name = "R_2__static_and_dynamic_score_according_to_input_dataset_size_n_cov_"+ str(n_best_covariates) + "_n_strong_interact_" + str(n_strong_interact) + ".png"
    #input_dataset_size = [len(X)]
    input_dataset_size = np.arange(start=100, stop=len(X), step=25).tolist()
    R2_score = []
    dynamic_score = []
    linear_R2_score = []
    static_score = []
    for dataset_size_to_consider in input_dataset_size: 
        acceptance_degree = math.floor(dataset_size_to_consider * acceptance_ratio) 
        print("--- Resetting the kernel matrix")
        R2_, dynamic_, linear_R2_ , static_score_ = get_dynamic_and_R2_score(phone_name, energy_gap, number_of_neighbour,
                    dataset_size_to_consider,  X_user_friendly, X, X_dict, y, energy_array, workload_array,
                 input_format, consider_exact_values_of_frequency, X_format_in_model, 
                  populate_inputs_to_considere_thread_combinations_on_same_socket, 
                 maximum_number_of_combination, one_hot_encoding_of_frequency,standartize_inputs,
                 search_ridge_coeff, search_strategy, number_of_iteration_during_exploration, l_o_o_on_alpha_exploration, ltolerance,lambda_min, max_iterations,sequential_gap,
                 dichotomic_progression_ratio, 
                 generate_plots, result_summary_csv_file_name, alpha, 
                 repeat_experiments, n_best_covariates, n_strong_interact, acceptance_degree )
        
        R2_score.append(R2_)
        dynamic_score.append(dynamic_)
        linear_R2_score.append(linear_R2_)
        static_score.append(static_score_)
    print("--- Plotting R2 score and dynamic score.")
    print("--- List of input dataset sizes: ", input_dataset_size)
    print("--- R2 score  list:", R2_score )
    print("--- Dynamic score list:", dynamic_score )
    print("--- linear R2 score list:", linear_R2_score )
    print("--- static score list:", static_score )
    paper_fontsize = 16
    static_score_plot = plt.figure()
    plt.plot(input_dataset_size, R2_score ,  color='black', marker='o', linestyle='-' , label =  r'$R^2$' + " score")
    plt.plot(input_dataset_size, static_score ,  color='#800000', marker='o', linestyle='-', label = "static score" )
    plt.plot(input_dataset_size, dynamic_score ,  color='orange', marker='o', linestyle='-', label = "dynamic score" )
    
    plt.title( "FitsAll R2 and dynamic scores according to \n the input dataset size N", fontsize = paper_fontsize)
    plt.legend(loc="upper left",  fontsize = paper_fontsize)
    plt.yticks(fontsize=paper_fontsize)
    #plt.xticks(fontsize=paper_fontsize)
    plt.xlabel( "Dataset size" ,  fontsize = paper_fontsize )
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(paper_fontsize)
    #plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()       
    plt.savefig("finding_best_input_dataset_size/"+ output_plot_name )
    plt.clf()
    plt.cla()
    plt.close()

    print("plot produced in file ", "finding_best_input_dataset_size/" + output_plot_name)
    if (output_plot_name != ""):   
        output_file_path = "finding_best_input_dataset_size/" + output_plot_name + "sorted_dynamic.csv" 
        input_size_R2_dynamic_score_array = [ [input_size_, dynamic_, R_2] for input_size_, dynamic_, R_2 in zip(input_dataset_size, dynamic_score, R2_score)]
        with open(output_file_path,'w') as file:
            file.write("input_data_set, R2 score, dynamic_score\n")  
            sorted_input_size_R2_dynamic_score_array= sorted(input_size_R2_dynamic_score_array, key=lambda kv: kv[1],  reverse=True) 
            for triplet in sorted_input_size_R2_dynamic_score_array:
                file.write(str(triplet[0]) + ", "  + str(triplet[1]) + ", "  + str(triplet[2]) + "\n")
        output_file_path =  "finding_best_input_dataset_size/" + output_plot_name + "sorted_R2.csv" 
        with open(output_file_path,'w') as file:
            file.write("input_data_set, R2 score, dynamic_score\n")
            sorted_input_size_R2_dynamic_score_array = sorted(input_size_R2_dynamic_score_array, key=lambda kv: kv[2],  reverse=True) 
            for triplet in sorted_input_size_R2_dynamic_score_array:
                file.write(str(triplet[0]) + ", "  + str(triplet[1]) + ", "  + str(triplet[2]) + "\n")

    ### plotting Ordinary least square, and kernel ridge according to dataset_sised
    output_plot_name = "Ordinary_linear_and_kernel_ridge_R_2_scores_according_to_input_dataset_size.png"
    static_score_plot = plt.figure()
    plt.plot(input_dataset_size, R2_score ,  color='black', marker='o', linestyle='-' , label =  "kernel ridge " +  r'$R^2$' + " score")
    plt.plot(input_dataset_size, linear_R2_score ,  color='gray', marker='o', linestyle='-', label =  "Ordinary linear model " +  r'$R^2$' + " score" )
  
    plt.title( "Kernel ridge and ordinary linear models R2 scores \n according to the input dataset size N", fontsize = paper_fontsize)
    plt.legend(loc="lower right",  fontsize = paper_fontsize)
    plt.yticks(fontsize=paper_fontsize)
    #plt.xticks(fontsize=paper_fontsize)
    plt.xlabel( "Dataset size" ,  fontsize = paper_fontsize )
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(paper_fontsize)
    #plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()       
    plt.savefig("finding_best_input_dataset_size/"+ output_plot_name )
    plt.clf()
    plt.cla()
    plt.close()


    #### Plot realized to reduce the paper size
    paper_fontsize = 20
    output_plot_name = "R_2__static_and_dynamic_scores___Ordinary_linear_and_kernel_ridge_R_2_scores.png"
    fig, ((R_2__static_and_dynamic_scores, Ordinary_linear_and_kernel_ridge_R_2_scores) ) = plt.subplots(nrows= 1, ncols = 2, sharex = True, figsize=(15, 7))
   
    R_2__static_and_dynamic_scores.plot(input_dataset_size, R2_score ,  color='black', marker='o', linestyle='-' , label =  r'$R^2$' + " score")
    R_2__static_and_dynamic_scores.plot(input_dataset_size, static_score ,  color='#800000', marker='o', linestyle='-', label = "static score" )
    R_2__static_and_dynamic_scores.plot(input_dataset_size, dynamic_score ,  color='orange', marker='o', linestyle='-', label = "dynamic score" )
    R_2__static_and_dynamic_scores.set_xlabel( "Dataset size" ,  fontsize = paper_fontsize )
    R_2__static_and_dynamic_scores.tick_params(axis='x', which='major' , labelsize= paper_fontsize)
    R_2__static_and_dynamic_scores.tick_params(axis='y', which='major' , labelsize= paper_fontsize)
    R_2__static_and_dynamic_scores.tick_params(size=8)
    R_2__static_and_dynamic_scores.legend(loc = 'upper left', prop={'size': 18})
    R_2__static_and_dynamic_scores.set_title("FitsAll R2 and dynamic scores, \n according to the input dataset size N", fontsize = paper_fontsize)
    

    Ordinary_linear_and_kernel_ridge_R_2_scores.plot(input_dataset_size, R2_score ,  color='black', marker='o', linestyle='-' , label =  "kernel ridge " +  r'$R^2$' + " score")
    Ordinary_linear_and_kernel_ridge_R_2_scores.plot(input_dataset_size, linear_R2_score ,  color='gray', marker='o', linestyle='-', label =  "Ordinary linear model " +  r'$R^2$' + " score" )
  
   
    Ordinary_linear_and_kernel_ridge_R_2_scores.tick_params(axis='x', which='major' , labelsize= paper_fontsize)
    Ordinary_linear_and_kernel_ridge_R_2_scores.tick_params(axis='y', which='major' , labelsize= paper_fontsize)
    Ordinary_linear_and_kernel_ridge_R_2_scores.set_xlabel("Dataset size" ,  fontsize = paper_fontsize)
    Ordinary_linear_and_kernel_ridge_R_2_scores.tick_params(size=8)
    Ordinary_linear_and_kernel_ridge_R_2_scores.set_title("Kernel ridge and Ord. linear models R2 scores \n according to the input dataset size N", fontsize = paper_fontsize)

     
    plt.savefig("finding_best_input_dataset_size/"+ output_plot_name )
    plt.clf()
    plt.cla()
    plt.close()


    
    print("--- Dynamic score: "+ str(dynamic_score) )
    print("--- R2 score: ", R2_score)
    print("--- Linear R2 score: ", linear_R2_score)
    print("--- Static score: ", static_score)
end = time.time()
#Subtract Start Time from The End Time
total_time = end - start
print("--- Total execution time: "+ str(total_time) + " seconds = " + repr(total_time/60) + " mins")







