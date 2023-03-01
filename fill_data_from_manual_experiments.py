from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import utils_functions as utils


powertool_working_directory = "/home/patrick/opportunistic_tasks_android/power_tool_results/"

# google_pixel_varying_the_number_of_thread/ (if there is no mention of google pixel format it is the generic format)
X_user_friendly_google_pixel_4a_5g_varying_number_of_threads = utils.read_configuration_in_user_frendly_format(
    powertool_working_directory + "google_pixel_varying_the_number_of_thread/machine_learning_datas/configurations_user_friendly.txt")
print (" ***** Configurations user friendly of google_pixel_4a_5g_varying_number_of_threads: \n", X_user_friendly_google_pixel_4a_5g_varying_number_of_threads)

X_google_pixel_4a_5g_varying_number_of_threads_generic_format = utils.read_configuration_X(
    powertool_working_directory + "google_pixel_varying_the_number_of_thread/machine_learning_datas/configurations_generic_format.txt")
print ("***** Configurations formatted of google_pixel_4a_5g_varying_number_of_threads: \n", X_google_pixel_4a_5g_varying_number_of_threads_generic_format)
X_google_pixel_4a_5g_varying_number_of_threads_generic_format_exact_freq = utils.read_configuration_X(
    powertool_working_directory + "google_pixel_varying_the_number_of_thread/machine_learning_datas/configurations_generic_format_exact_frequency_values.txt")
print ("***** Configurations formatted of google_pixel_4a_5g_varying_number_of_threads: \n", X_google_pixel_4a_5g_varying_number_of_threads_generic_format_exact_freq)


X_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format = utils.read_configuration_X(
    powertool_working_directory + "google_pixel_varying_the_number_of_thread/machine_learning_datas/configurations_google_pixel_format.txt")
print ("***** Configurations formatted of google_pixel_4a_5g_varying_number_of_threads: \n", X_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format)
X_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format_exact_freq = utils.read_configuration_X(
    powertool_working_directory + "google_pixel_varying_the_number_of_thread/machine_learning_datas/configurations_google_pixel_format_exact_frequency_values.txt")
print ("***** Configurations formatted of google_pixel_4a_5g_varying_number_of_threads: \n", X_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format_exact_freq)

X_dict_google_pixel_4a_5g_varying_number_of_threads_generic_format = utils.create_dictionnary_for_X(
    X_user_friendly_google_pixel_4a_5g_varying_number_of_threads, X_google_pixel_4a_5g_varying_number_of_threads_generic_format)
print ("***** Configurations dictionnary of google_pixel_4a_5g_varying_number_of_threads_generic_format: \n", X_dict_google_pixel_4a_5g_varying_number_of_threads_generic_format)
X_dict_google_pixel_4a_5g_varying_number_of_threads_generic_format_exact_freq = utils.create_dictionnary_for_X(
    X_user_friendly_google_pixel_4a_5g_varying_number_of_threads, X_google_pixel_4a_5g_varying_number_of_threads_generic_format_exact_freq)
print ("***** Configurations dictionnary of google_pixel_4a_5g_varying_number_of_threads_generic_format: \n", X_dict_google_pixel_4a_5g_varying_number_of_threads_generic_format)
X_dict_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format = utils.create_dictionnary_for_X(
    X_user_friendly_google_pixel_4a_5g_varying_number_of_threads, X_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format)
print ("***** Configurations dictionnary of google_pixel_4a_5g_varying_number_of_threads_google_pixel_format: \n", X_dict_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format)
X_dict_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format_exact_freq = utils.create_dictionnary_for_X(
    X_user_friendly_google_pixel_4a_5g_varying_number_of_threads, X_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format_exact_freq)
print ("***** Configurations dictionnary of google_pixel_4a_5g_varying_number_of_threads_google_pixel_format: \n", X_dict_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format_exact_freq)


y_google_pixel_4a_5g_varying_number_of_threads = utils.read_ratio_energy_by_workload(
    powertool_working_directory + "google_pixel_varying_the_number_of_thread/machine_learning_datas/ratio_energy_by_worload.txt")
print("***** Ratio energy by wokload of google_pixel_4a_5g_varying_number_of_threads: \n", y_google_pixel_4a_5g_varying_number_of_threads)




# google_pixel_4a_5g_varying_frequency
X_user_friendly_google_pixel_4a_5g_varying_frequency = utils.read_configuration_in_user_frendly_format(
    powertool_working_directory + "google_pixel_4a_5g_varying_frequency/machine_learning_datas/configurations_user_friendly.txt")
print ("***** Configurations user friendly of google_pixel_4a_5g_varying_frequency: \n", X_user_friendly_google_pixel_4a_5g_varying_frequency)


X_google_pixel_4a_5g_varying_frequency_generic_format = utils.read_configuration_X(
    powertool_working_directory + "google_pixel_4a_5g_varying_frequency/machine_learning_datas/configurations_generic_format.txt")
print ("***** Configurations formatted of google_pixel_4a_5g_varying_frequency: \n", X_google_pixel_4a_5g_varying_frequency_generic_format)
X_google_pixel_4a_5g_varying_frequency_generic_format_exact_freq = utils.read_configuration_X(
    powertool_working_directory + "google_pixel_4a_5g_varying_frequency/machine_learning_datas/configurations_generic_format_exact_frequency_values.txt")
print ("***** Configurations formatted of google_pixel_4a_5g_varying_frequency: \n", X_google_pixel_4a_5g_varying_frequency_generic_format)



X_google_pixel_4a_5g_varying_frequency_google_pixel_format = utils.read_configuration_X(
    powertool_working_directory + "google_pixel_4a_5g_varying_frequency/machine_learning_datas/configurations_google_pixel_format.txt")
print ("***** Configurations formatted of google_pixel_4a_5g_varying_frequency: \n", X_google_pixel_4a_5g_varying_frequency_google_pixel_format)
X_google_pixel_4a_5g_varying_frequency_google_pixel_format_exact_freq = utils.read_configuration_X(
    powertool_working_directory + "google_pixel_4a_5g_varying_frequency/machine_learning_datas/configurations_google_pixel_format_exact_frequency_values.txt")
print ("***** Configurations formatted of google_pixel_4a_5g_varying_frequency: \n", X_google_pixel_4a_5g_varying_frequency_google_pixel_format)





X_dict_google_pixel_4a_5g_varying_frequency_generic_format = utils.create_dictionnary_for_X(
    X_user_friendly_google_pixel_4a_5g_varying_frequency, X_google_pixel_4a_5g_varying_frequency_generic_format)
print ("***** Configurations dictionnary of google_pixel_4a_5g_varying_frequency_generic_format: \n", X_dict_google_pixel_4a_5g_varying_frequency_generic_format)
X_dict_google_pixel_4a_5g_varying_frequency_generic_format_exact_freq = utils.create_dictionnary_for_X(
    X_user_friendly_google_pixel_4a_5g_varying_frequency, X_google_pixel_4a_5g_varying_frequency_generic_format_exact_freq)
print ("***** Configurations dictionnary of google_pixel_4a_5g_varying_frequency_generic_format_exact_freq: \n", X_dict_google_pixel_4a_5g_varying_frequency_generic_format_exact_freq)
X_dict_google_pixel_4a_5g_varying_frequency_google_pixel_format = utils.create_dictionnary_for_X(
    X_user_friendly_google_pixel_4a_5g_varying_frequency, X_google_pixel_4a_5g_varying_frequency_google_pixel_format)
print ("***** Configurations dictionnary of google_pixel_4a_5g_varying_frequency_google_pixel_format: \n", X_dict_google_pixel_4a_5g_varying_frequency_google_pixel_format)
X_dict_google_pixel_4a_5g_varying_frequency_google_pixel_format_exact_freq = utils.create_dictionnary_for_X(
    X_user_friendly_google_pixel_4a_5g_varying_frequency, X_google_pixel_4a_5g_varying_frequency_google_pixel_format_exact_freq)
print ("***** Configurations dictionnary of google_pixel_4a_5g_varying_frequency_google_pixel_format_exact_freq: \n", X_dict_google_pixel_4a_5g_varying_frequency_google_pixel_format_exact_freq)







y_google_pixel_4a_5g_varying_frequency = utils.read_ratio_energy_by_workload(
    powertool_working_directory + "google_pixel_4a_5g_varying_frequency/machine_learning_datas/ratio_energy_by_worload.txt")
print("***** Ratio energy by wokload of google_pixel_4a_5g_varying_frequency: \n", y_google_pixel_4a_5g_varying_frequency)




# to process with "powertool_working_directory +""  and "machine_learning_datas"



# samsung_galaxy_s8_varying_number_of_threads
X_user_friendly_samsung_galaxy_s8_varying_number_of_threads = utils.read_configuration_in_user_frendly_format(
    "data_samsung_galaxy_s8_varying_number_of_threads/configurations_user_friendly.txt")
print ("***** Configurations user friendly of samsung_galaxy_s8_varying_number_of_threads: \n", X_user_friendly_samsung_galaxy_s8_varying_number_of_threads)
X_samsung_galaxy_s8_varying_number_of_threads = utils.read_configuration_X(
    "data_samsung_galaxy_s8_varying_number_of_threads/configurations_generic_format.txt")
print ("***** Configurations formatted of samsung_galaxy_s8_varying_number_of_threads: \n", X_samsung_galaxy_s8_varying_number_of_threads)
X_dict_samsung_galaxy_s8_varying_number_of_threads = utils.create_dictionnary_for_X(
    X_user_friendly_samsung_galaxy_s8_varying_number_of_threads, X_samsung_galaxy_s8_varying_number_of_threads)
print ("***** Configurations dictionnary of samsung_galaxy_s8_varying_number_of_threads: \n", X_dict_samsung_galaxy_s8_varying_number_of_threads)
y_samsung_galaxy_s8_varying_number_of_threads = utils.read_ratio_energy_by_workload(
    "data_samsung_galaxy_s8_varying_number_of_threads/ratio_energy_by_worload.txt")
print("***** Ratio energy by wokload of samsung_galaxy_s8_varying_number_of_threads: \n", y_samsung_galaxy_s8_varying_number_of_threads)

#samsung_galaxy_s8_varying_frequency
X_user_friendly_samsung_galaxy_s8_varying_frequency = utils.read_configuration_in_user_frendly_format(
    "data_samsung_galaxy_s8_varying_frequency/configurations_user_friendly.txt")
print ("***** Configurations user friendly of samsung_galaxy_s8_varying_frequency: \n", X_user_friendly_samsung_galaxy_s8_varying_frequency)
X_samsung_galaxy_s8_varying_frequency = utils.read_configuration_X(
    "data_samsung_galaxy_s8_varying_frequency/configurations_generic_format.txt")
print ("***** Configurations formatted of samsung_galaxy_s8_varying_frequency: \n", X_samsung_galaxy_s8_varying_frequency)
X_dict_samsung_galaxy_s8_varying_frequency = utils.create_dictionnary_for_X(
    X_user_friendly_samsung_galaxy_s8_varying_frequency, X_samsung_galaxy_s8_varying_frequency)
print ("***** Configurations dictionnary of samsung_galaxy_s8_varying_frequency: \n", X_dict_samsung_galaxy_s8_varying_frequency)
y_samsung_galaxy_s8_varying_frequency = utils.read_ratio_energy_by_workload(
    "data_samsung_galaxy_s8_varying_frequency/ratio_energy_by_worload.txt")
print("***** Ratio energy by wokload of samsung_galaxy_s8_varying_frequency: \n", y_samsung_galaxy_s8_varying_frequency)



########### Add generic function to pick up good dictionnary and goog X parameter depending on user parmater. 






# filling X
def get_X(phone_name, input_format, consider_exact_values_of_frequency = False):
    if phone_name == "google_pixel_4a_5g" and input_format == "google_pixel_4a_5g_format" and consider_exact_values_of_frequency:
        return  X_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format_exact_freq + X_google_pixel_4a_5g_varying_frequency_google_pixel_format_exact_freq

    elif phone_name == "google_pixel_4a_5g" and input_format == "google_pixel_4a_5g_format" and not consider_exact_values_of_frequency:
        return X_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format + X_google_pixel_4a_5g_varying_frequency_google_pixel_format
    
        
    elif phone_name == "google_pixel_4a_5g" and input_format == "generic" and  consider_exact_values_of_frequency:
        return X_google_pixel_4a_5g_varying_number_of_threads_generic_format_exact_freq + \
           X_google_pixel_4a_5g_varying_frequency_generic_format_exact_freq 
     

    elif phone_name == "google_pixel_4a_5g" and input_format == "generic" and not consider_exact_values_of_frequency:
        return X_google_pixel_4a_5g_varying_number_of_threads_generic_format + \
           X_google_pixel_4a_5g_varying_frequency_generic_format  
    
    elif phone_name == "google_pixel_4a_5g" and input_format == "human_readable_format" :
        return X_user_friendly_google_pixel_4a_5g_varying_number_of_threads + \
                        X_user_friendly_google_pixel_4a_5g_varying_frequency  

     
    #...

# filling dictionnary
def get_X_dict(phone_name, input_format, consider_exact_values_of_frequency):  
    if phone_name == "google_pixel_4a_5g" and input_format == "google_pixel_4a_5g_format" and consider_exact_values_of_frequency:
        X_dict = X_dict_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format_exact_freq.copy()
        X_dict.update(X_dict_google_pixel_4a_5g_varying_frequency_google_pixel_format_exact_freq) 
        return X_dict

    elif phone_name == "google_pixel_4a_5g" and input_format == "google_pixel_4a_5g_format" and not consider_exact_values_of_frequency:
        X_dict = X_dict_google_pixel_4a_5g_varying_number_of_threads_google_pixel_format.copy()
        X_dict.update(X_dict_google_pixel_4a_5g_varying_frequency_google_pixel_format) 
        return X_dict
    
    elif phone_name == "google_pixel_4a_5g" and input_format == "generic" and consider_exact_values_of_frequency:
        X_dict = X_dict_google_pixel_4a_5g_varying_number_of_threads_generic_format_exact_freq.copy()
        X_dict.update(X_dict_google_pixel_4a_5g_varying_frequency_generic_format_exact_freq) 
        return X_dict
    elif phone_name == "google_pixel_4a_5g" and input_format == "generic" and not consider_exact_values_of_frequency:
        X_dict = X_dict_google_pixel_4a_5g_varying_number_of_threads_generic_format.copy()
        X_dict.update(X_dict_google_pixel_4a_5g_varying_frequency_generic_format) 
        return X_dict


# filling y
def get_y(phone_name):
    if (phone_name == "google_pixel_4a_5g"):
         return y_google_pixel_4a_5g_varying_number_of_threads + \
            y_google_pixel_4a_5g_varying_frequency 
    #...  











"""
#### to process 

X_user_friendly_samsung_galaxy_s8 =  X_user_friendly_samsung_galaxy_s8_varying_number_of_threads + \
    X_user_friendly_samsung_galaxy_s8_varying_frequency

X_samsung_galaxy_s8 =  X_samsung_galaxy_s8_varying_number_of_threads + \
    X_samsung_galaxy_s8_varying_frequency

X_dict_samsung_galaxy_s8 =  X_dict_samsung_galaxy_s8_varying_number_of_threads.copy()
X_dict_samsung_galaxy_s8.update(X_dict_samsung_galaxy_s8_varying_frequency)

y_samsung_galaxy_s8 = y_samsung_galaxy_s8_varying_number_of_threads + \
    y_samsung_galaxy_s8_varying_frequency





#### Taking into account the real value of frequency 
# for a particular dictionnary this function transform the observation X[i], to reflect exact values of frequency
# possible dictionnaries can be: 
#  1 - dictionnary of experiment on google pixel 4a 5g on little socket, 
#          0 ->  0 
#          1 -> 576000   (it is the minimum usable frequency on little core on google pixel 4a 5g)
#          2 -> 1363200  (it is the mid frequency on little core on google pixel 4a 5g)
#          3 -> 1804800  (it is the max frequency on little core on google pixel 4a 5g)
freq_dict_of_little_socket_google_pixel = { 0: 0, 1: 576000,  2: 1363200,  3: 1804800  }
#  2 - dictionnary of experiments on google pixel 4a 5g  on medium socket, 
#          0 -> 0
#          1 -> 652800   (it is the minimum usable frequency on big core on google pixel 4a 5g)
#          2 -> 1478400  (it is the mid frequency on big core on google pixel 4a 5g)
#          3 -> 2208000  (it is the max frequency on big core on google pixel 4a 5g )
freq_dict_of_medim_socket_google_pixel = { 0: 0, 1: 652800,  2: 1478400,  3: 2208000  }
#### Before continue I put there a big warning because, I need to reconsider the fact that google pixel à three types of cores (medium, little, big cores)
####
#  3 - dictionnary of experiment on google pixel 4a 5g on big socket, 
#          0 ->  0 
#          1 -> 806400   (it is the minimum usable frequency on big core on google pixel 4a 5g)
#          2 -> 1766400   (it is the mid frequency on big core on google pixel 4a 5g)
#          3 -> 2400000  (it is the max frequency on big core on google pixel 4a 5g )
freq_dict_of_big_socket_google_pixel = { 0: 0, 1: 806400,  2: 1766400,  3: 2400000  }
#  1 - dictionnary of experiment on samsung galaxy s8  on little socket, 
#          0 ->  0 
#          1 -> 598000   (it is the minimum usable frequency on little core on samsung galaxy s8 )
#          2 -> 1248000  (it is the mid frequency on little core on samsung galaxy s8 )
#          3 -> 1690000  (it is the max frequency on little core on samsung galaxy s8 )
freq_dict_of_little_socket_samsung_galaxy_s8 = { 0: 0, 1: 598000,  2: 1248000,  3: 1690000  }
#  2 - dictionnary of experiments on samsung galaxy s8  on medium socket, 
#          0 ->  0 / because the configuration of samsung galaxy s8 do not have medium core. 
freq_dict_of_medim_socket_samsung_galaxy_s8 = { 0: 0, 1: 0,  2: 0,  3: 0  }
#  3 - dictionnary of experiment on samsung galaxy s8  on big socket, 
#          0 ->  0 
#          1 -> 741000   (it is the minimum usable frequency on big core on samsung galaxy s8 )
#          2 -> 1469000  (it is the mid frequency on big core on samsung galaxy s8 )
#          3 -> 2314000  (it is the max frequency on big core on samsung galaxy s8 )
freq_dict_of_big_socket_samsung_galaxy_s8 = { 0: 0, 1: 741000,  2: 1469000,  3: 2314000  }

def trans_form_X_to_have_real_values_of_frequency(X, freq_dict_of_little_socket , freq_dict_of_medium_socket, freq_dict_of_big_socket):
    result = []
    for x in X:
        temp_x = [] 
        for i in range(0,6):
            temp_x.append(freq_dict_of_little_socket[x[i]])
        for i in range(6,9):
            temp_x.append(freq_dict_of_medium_socket[x[i]])
        for i in range(9,13):
            temp_x.append(freq_dict_of_big_socket[x[i]])
        result.append(temp_x)
    return result
