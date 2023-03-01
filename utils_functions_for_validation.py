# this file is for the validation of the kernel ridge model


# 1 - It first does estimation or the best configuration by the kernel ridge model
# the function write_sorted_estimations_of_configurations is written for it
# it first generate all possible configuration with the same strategy as the file can_be_reused/generate_really_tested_configurations_for_[...].py
# secondly, based on the kernel ridge model parameter computed, it estimate the efficiency of each configurations and 
# sort it 
# write it in the output file.
# In the first use case of this script,  we do not considere the level 4 frequency level (where the governor is not touched),
# because we whant to analyse very first experiments results. 

# 2 - other validation function can be written, but maybe in the future. 
#   a) for database creation create_database(X_user_friendly, X, y, energy_array, workload_array, X_test, X_train, Y_test, X_train)
import random
from itertools import product
import utils_functions as utils
import matplotlib.pyplot as plt 
import sqlite3
# NOTE: to integretate the situation where the governor of the socket is not touched, frequency level can take tbe value of 4
# in this case put this variable (consider_free_frequency) to 1 before continue
consider_free_frequency = 0
# To generate only configurations where there is at least one socket with default governor
only_with_frequency_free = False

if consider_free_frequency == 1: 
    max_frequency_level = 4
else: 
    max_frequency_level = 3
# TODO In some comment this value is 3 because we used 3 frequency levels at the beginning of the project (need to modify it later!!!)
#

def convert_in_base(decimal_value, base, length = 8):
    # convert a number in base 10 to its corresponding value in base Y"
    #For exemple the configuration   0, 3, 3, 3, 0, 0, 0, 2
    #will have the base notation  2  0  1  1  1  0  0  0  2
    # return the result as a numpy array
    print (" --- converting " + str(decimal_value) + " in base " + str(base))

    # first
    list_of_rest=[]
    if (decimal_value < base):
        list_of_rest.append(decimal_value)
        for index in range(len(list_of_rest), length):
            list_of_rest.append(0)
        print (" --- result ", list(reversed(list_of_rest)))
        return  list(reversed(list_of_rest))
    
    
    
    last_result = decimal_value
    while last_result >= base:    
        print (" --- appending " + str(int(last_result) % int(base)) + " to result")
        list_of_rest.append(int(last_result) % int(base))
        last_result = last_result/base
    print (" --- appending " + str(int(last_result)) + " to result")
    list_of_rest.append(int(last_result))

    for index in range(len(list_of_rest), length):
        list_of_rest.append(0)

    print (" --- result ", list(reversed(list_of_rest)))

    return list(reversed(list_of_rest))



def convert_from_triplet_to_base_Y(triplet_of_decimal_value):
    # convert a triplet of decimal numbers from decimal base to  base 3, 2 and 4 respectively "
    # return a list of concatenated arrays, each array is the representation of decimal number in the bases listed above
    # For exemple the triplet  (3, 4, 3)
    #  the input array  will have the base notation (( 3 in base max_frequency_level on one bits), (4 in base 2 on six bits )(3 in base max_frequency_level+1 on tow bits))
    # this function will just return an array
    print (" --- converting " + repr(triplet_of_decimal_value) + "  to base Y  (  base 3 on 2 bits, 2 on 6 bits and 4 on 2 bits)")

    result_1 = convert_in_base(triplet_of_decimal_value[0], max_frequency_level, 1)
    result_2 = convert_in_base(triplet_of_decimal_value[1], 2, 6)
    result_3 = convert_in_base(triplet_of_decimal_value[2], max_frequency_level+1, 2)
    result = result_1 + result_2 + result_3

    print (" --- Result = ",  result)
    return result

"""
def convert_from_base_Y_to_configuration_notation(array_in_base_Y):
    # convert a base Y notation to the configuration notation 
    # this code will check the value of the first element of the array and apply the corresponding frequency level on next sixth element of the array given as parameter
    
    print (" --- Converting  base Y number " +  repr(array_in_base_Y) + " to configuration notation" )

    frequency_level = array_in_base_Y[0] + 1
    result = []
    
    for index in range(1, 7):
        if array_in_base_Y[index] == 1:
            result.append(frequency_level)
        else:
            result.append(0)
    for index in range(7, len(array_in_base_Y)):
        result.append(array_in_base_Y[index])
        
    print (" --- Result = ",  result)
    return result
"""




def convert_array_from_base(array_of_values, base):
    # convert a number  in base "base" to its corresponding value  in base 10 
    # return the result as a numpy array 
    print (" --- Converting array " +  repr(array_of_values) + " from base " + str(base) + " to decimal" )
    result = 0
    for index in range(0, len(array_of_values)):
        j = len(array_of_values) - 1   - index
        result = result +  array_of_values[j] * ( base ** index )

    print (" --- result ", result)

    return result



def convert_from_triplet_to_configuration(triplet):
    return convert_from_base_Y_to_configuration_notation(convert_from_triplet_to_base_Y(triplet))


def substract_already_tested_configuration (configurations_input_file, configuration_to_subtract, configuration_output_file):
    ## This function substract some configurations from a list of configurations.
    #  The configurations passed as input are the one generated by function above
    # The configuration to substract are the one already tested 
    # They are readed by these function  in a summary file generated by the automatization script 
    # As a result we should have a list of configuration with the same format as the list passed as input. 
    
    print("--- Raeding the configuration to subtract file ")
    list_of_configuration_to_substract=[]
    file_to_subtract = open(configuration_to_subtract, 'r')
    header=1
    while True:
        line = file_to_subtract.readline()
        if not line:
            break
        datas_in_line = line.split(",")
        configuration_as_string=datas_in_line[0]
        if(header == 0):
            print("--- Reading next configuration to substract", configuration_as_string)
            list_of_configuration_to_substract.append(configuration_as_string)
        if (header == 1):
            header=0
      
    file_to_subtract.close()

    print("--- Writing the configuration output file ")   
    with open(configuration_output_file,'w') as file:
        file.write("configurations,google pixel format\n")
        input_file = open(configurations_input_file, 'r') 
        header=1
        while True:
            line = input_file.readline()
            if not line:
                break
            if(header == 1):
                header=0
                continue
            
            datas_in_line = line.split(",")
            current_configuration_as_string=datas_in_line[0]
            print ("--- Testing this configuration ", current_configuration_as_string)
            if (current_configuration_as_string in list_of_configuration_to_substract ):
                print("--- We don't add this configuration ",current_configuration_as_string)
            else:
                print ("--- We add this configuration")
                file.write(line)         
        input_file.close()


def get_configurations_from_summary_file(summary_file_path):
    print (" --- Getting data from file ", summary_file_path)
    result = []
    summary_file = open(summary_file_path, 'r') 
    header=1
    while True:
        line = summary_file.readline()
        if not line:
            break
        if(header == 1):
            header=0
            continue
        print("--- Reading the line :", line)
        line_data = line.split(",")  
        current_configuration = line_data[3]
        current_configuration = (current_configuration[1:-1]).replace(" ", "").split("-")
        print ("arrays of value as string : ", current_configuration)
        X_line = [int(numeric_string) for numeric_string in current_configuration]
        print("resulted X configuration : ", X_line)
        result.append(X_line) 
    return result           
            



import os
def get_configurations_from_summary_folder( summary_folder_path ):
    total_result_list = []
    list_of_files = {}
    print (" --- Getting data from folder ", summary_folder_path)
    for (dirpath, dirnames, filenames) in os.walk(summary_folder_path):
        for filename in filenames:
            #if filename.endswith('.html'): 
            #list_of_files[filename] = os.sep.join([dirpath, filename])
            current_result_list = get_configurations_from_summary_file(os.sep.join([dirpath, filename]))
            total_result_list = total_result_list + current_result_list
    return total_result_list


import itertools
import numpy as np
def array_in_list_of_array(val, list_of_array):
    place = 0
    for curr in list_of_array:
        if (np.array_equal(curr, val)):
            return place
        place = place + 1

    return -1


def four_is_in_configuration(i):
    configuration_as_array = convert_from_triplet_to_configuration(i)
    for i in configuration_as_array:
        if i == 4: 
            return True
    return False




def convert_configuration_to_suitable_format_for_estimation(configuration, input_format, X_format_in_model, exact_values_of_frequency):
    result = []
    if (input_format == "google_pixel_4a_5g_format"): # or  can be"samsung_galaxy_s8_format", or "generic"
        if (not exact_values_of_frequency):
            if(X_format_in_model == "base_Y"):
                # In the case where we cannot longer set different values of frequencies to core on the same socket; 
                # variables are no longer independant, 
                # So we introduce another notation factorizing the not nul frequency level to the first newly introduced position
                result = utils.convert_from_configuration_to_base_Y(configuration)
                   
            elif (X_format_in_model == "base_Y_N_on_socket"):      
                result = utils.convert_from_configuration_to_base_Y_N_on_socket(configuration)
       
            elif(X_format_in_model == "base_Y_F"):
                # More than Base_Y,  foreach socket another variable has been introduced concerning the fact that the frequency of the socket is modify or no
                result = utils.convert_from_configuration_to_base_Y_F(configuration)
            elif(X_format_in_model == "base_Y_F_N_on_socket"):
                # More than Base_Y,  foreach socket another variable has been introduced concerning the fact that the frequency of the socket is modify or no
                result = utils.convert_from_configuration_to_base_Y_F_N_on_socket(configuration)
    elif (input_format == "samsung_galaxy_s8_format"):
        if (not exact_values_of_frequency):
            if(X_format_in_model == "base_Y"):
                # In the case where we cannot longer set different values of frequencies to core on the same socket; 
                # variables are no longer independant, 
                # So we introduce another notation factorizing the not nul frequency level to the first newly introduced position
            
                result = utils.convert_from_configuration_to_base_Y(configuration, input_format)
                #print("--- Special case : resulted X configuration in base Y format: ", X_line)
              
            elif (X_format_in_model == "base_Y_N_on_socket"):
                
                result = utils.convert_from_configuration_to_base_Y_N_on_socket(configuration, input_format)
               
    return result




def convert_from_base_Y_to_configuration_notation(array_in_base_Y, experiment_format = "google_pixel_4a_5g_format"):
    # convert a base Y notation to the configuration notation 
    # this code will check the value of the first element of the array and apply the corresponding frequency level on next sixth element of the array given as parameter
    # for samsung it is the same principle for both sockets
    if experiment_format == "google_pixel_4a_5g_format":
            
        
        print (" --- Converting  base Y number " +  repr(array_in_base_Y) + " to google pixel configuration notation" )

        frequency_level = array_in_base_Y[0] + 1
        result = []
        
        for index in range(1, 7):
            if array_in_base_Y[index] == 1:
                result.append(frequency_level)
            else:
                result.append(0)
        for index in range(7, len(array_in_base_Y)):
            result.append(array_in_base_Y[index])
            
        print (" --- Result = ",  result)
    elif experiment_format == "samsung_galaxy_s8_format":
        print (" --- Converting  base Y number " +  repr(array_in_base_Y) + " to samsung configuration notation" )

        frequency_level = array_in_base_Y[0] + 1
        result = []
        
        for index in range(1, 5):
            if array_in_base_Y[index] == 1:
                result.append(frequency_level)
            else:
                result.append(0)


        frequency_level = array_in_base_Y[5] + 1    

        for index in range(6, 10):
            if array_in_base_Y[index] == 1:
                result.append(frequency_level)
            else:
                result.append(0)

            
        print (" --- Result = ",  result)
    
    
    return result


def convert_from_model_format_to_configuration(model_configuration, experiment_format = "google_pixel_4a_5g_format", X_format_in_model = "base_Y", exact_values_of_frequency = False):
    # this function convert a configuration from model format to experiment format. 
    # the model format is the one used by the model and can be Base_Y, Base_Y_F...
    # the configuration like format is the one used during experiment, can be for the google pixel or for the samsung 
    result = []
    if (experiment_format == "google_pixel_4a_5g_format"): # or  can be"samsung_galaxy_s8_format", or "generic"
        if (not exact_values_of_frequency):
            if(X_format_in_model == "base_Y"):
                # In the case where we cannot longer set different values of frequencies to core on the same socket; 
                # variables are no longer independant, 
                # So we introduce another notation factorizing the not nul frequency level to the first newly introduced position
                result = convert_from_base_Y_to_configuration_notation(model_configuration)
                   
            elif (X_format_in_model == "base_Y_N_on_socket"):  
                # TODO    
                #result = utils.convert_from_base_Y_N_on_socket_to_configuration_notation(model_configuration)
                print("--- Not yet implemented")
            elif(X_format_in_model == "base_Y_F"):
                # TODO
                # More than Base_Y,  foreach socket another variable has been introduced concerning the fact that the frequency of the socket is modify or no
                #result = utils.convert_from_base_Y_F_to_configuration(model_configuration)
                print("--- Not yet implemented")
            elif(X_format_in_model == "base_Y_F_N_on_socket"):
                #TODO
                # More than Base_Y,  foreach socket another variable has been introduced concerning the fact that the frequency of the socket is modify or no
                
                #result = utils.convert_from_base_Y_F_N_on_socket_to_configuration(model_configuration)
                print("--- Not yet implemented")
    elif (experiment_format == "samsung_galaxy_s8_format"):
        if (not exact_values_of_frequency):
            if(X_format_in_model == "base_Y"):
                # In the case where we cannot longer set different values of frequencies to core on the same socket; 
                # variables are no longer independant, 
                # So we introduce another notation factorizing the not nul frequency level to the first newly introduced position
            
                result = convert_from_base_Y_to_configuration_notation(model_configuration, input_format)
                #print("--- Special case : resulted X configuration in base Y format: ", X_line)
              
            elif (X_format_in_model == "base_Y_N_on_socket"):
                # TODO
                #result = utils.convert_from_base_Y_N_on_socket_to_configuration_notation(model_configuration, input_format)
                print("--- Not yet implemented")
    return result


def convert_from_experiment_format_to_user_friendly(experiment_configuration, experiment_format):
    user_friendly = ""
    if experiment_format == "google_pixel_4a_5g_format":
        user_friendly=str(experiment_configuration)[1:-1].replace(" ", "")
        user_friendly = user_friendly[:11] + '-' + user_friendly[11+1:]
        user_friendly = user_friendly[:13] + '-' + user_friendly[13+1:]
        user_friendly = str(user_friendly).replace(",", "")
    else:
        print ("NOT YET IMPLEMENTED")    
    return user_friendly


def  get_and_write_sorted_estimations_of_configurations(number_of_combinaison, gauss_process, input_format, X_format_in_model, exact_values_of_frequency,
    output_file_path = ""):

    
    

    special_cases_as_triplets = list(product( range(0, max_frequency_level), [0], range(0, (max_frequency_level + 1)**2) )) # See Notation SPACIAL CASE in the introduction comment of the file
    all_combinaisons_as_triplets =  list(product( range(0, max_frequency_level), range(0, 2**6), range(0, (max_frequency_level + 1)**2) ))
    
    configurations_candidates_as_triplet = []
    print("--- Number of possible combinations ", len(all_combinaisons_as_triplets))
    for i in all_combinaisons_as_triplets:       
        if (only_with_frequency_free and four_is_in_configuration(i)):
            print("--- Four is in configuration : ", i)
            print("--- array format : ", convert_from_triplet_to_configuration(i))
            if  (array_in_list_of_array (i, special_cases_as_triplets) !=-1 ):  # in this configuration the little socket has all cores off 0, it can be [M, 0, N]
                                                                            # if it representant, the configuration [0,0, N] as triplet is not yet considered as candidate we add it
                if (  array_in_list_of_array( [0, 0, i[2]], configurations_candidates_as_triplet) == -1  ):
                    configurations_candidates_as_triplet.append([0, 0, i[2]])
                    print("--- Considering a new special case configuration: ", [i[0], 0, i[2]])
                    print("--- Adding the representant: ", [0, 0, i[2]])
            else :
                print (" --- We are not in the presence of a special case, we add configuration :", i)
                configurations_candidates_as_triplet.append(i)
        else :
            print("--- Four is NOT in configuration : ", i)
            print("--- array format : ", convert_from_triplet_to_configuration(i))  
            if  (array_in_list_of_array (i, special_cases_as_triplets) !=-1 ):  # in this configuration the little socket has all cores off 0, it can be [M, 0, N]
                                                                            # if it representant, the configuration [0,0, N] as triplet is not yet considered as candidate we add it
                if ( array_in_list_of_array( [0, 0, i[2]], configurations_candidates_as_triplet) == -1  ):
                    configurations_candidates_as_triplet.append([0, 0, i[2]])
                    print("--- Considering a new special case configuration: ", [i[0], 0, i[2]])
                    print("--- Adding the representant: ", [0, 0, i[2]])
               
            else :
                print (" --- We are not in the presence of a special case, we add configuration :", i)
                configurations_candidates_as_triplet.append(i)

    print("--- Number of configuration candidates ", len(configurations_candidates_as_triplet))

    #configurations_candidates_as_triplet_retained = random.sample(configurations_candidates_as_triplet,)
    configuration__google_pixel_format__model_format__efficiency=[]
    



    list_of_retained_configurations = []
    configuration_google_pixel_format__model_format__efficiency = []
    for decimal_triplet_retained in configurations_candidates_as_triplet:
        combination = convert_from_triplet_to_configuration(decimal_triplet_retained)
        list_of_retained_configurations.append(combination)
        

        combination_for_estimation = convert_configuration_to_suitable_format_for_estimation(combination, input_format, X_format_in_model, exact_values_of_frequency )
   


        predicted_y_as_array =  gauss_process.predict(np.asarray([combination_for_estimation]))

        configuration_google_pixel_format__model_format__efficiency.append([combination, combination_for_estimation, predicted_y_as_array[0]])
        sorted_configuration_google_pixel_format__model_format__efficiency = sorted(configuration_google_pixel_format__model_format__efficiency, key=lambda kv: kv[2],  reverse=True) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
                 
    number_of_configuration_written = 0
    if (output_file_path != ""):   
        with open(output_file_path,'w') as file:
            file.write("rank, configurations,google pixel format, model format, energy_efficiency estimated\n")
            for raw_result in sorted_configuration_google_pixel_format__model_format__efficiency:
            
                retained_combination = raw_result[0]
                print ("--- Retained value to add in file: " + str(retained_combination) + ", indice : ", decimal_triplet_retained)
                user_friendly= convert_from_experiment_format_to_user_friendly(retained_combination, input_format)

                raw_configuration = str(raw_result[0]).replace(",", "-")
                model_type_configuration = str(raw_result[1])[1:-1].replace(" ", "").replace(",", "-") 
                energy_efficiency = str(raw_result[2])
                number_str = str(number_of_configuration_written)
                print("--- Raw result :", raw_result)


                string_to_write = number_str + "," + user_friendly + "," +  raw_configuration  + "," + model_type_configuration + "," + energy_efficiency
                

            
                if number_of_configuration_written < number_of_combinaison:
                    file.write(string_to_write)
                    file.write('\n')
                    number_of_configuration_written = number_of_configuration_written +1
                else: 
                    break
    
  
                  

    print("--- Number of configurations to write = ", number_of_combinaison)
    print("--- Number of configurations written = ", number_of_configuration_written)
    print("--- Size of sorted efficiency", len(sorted_configuration_google_pixel_format__model_format__efficiency))
    print("--- Outpuf file = ", output_file_path)
    return sorted_configuration_google_pixel_format__model_format__efficiency


def create_database_and_tables(conn):

    
    conn.execute('''DROP TABLE IF EXISTS configuration_representations;''') 
    conn.execute('''CREATE TABLE configuration_representations  
            (configuration_id INTEGER PRIMARY KEY   AUTOINCREMENT  NOT NULL,
            user_friendly_format       VARCHAR(50),
            experiment_format          VARCHAR(50),
            base_Y_format       VARCHAR(50)
            );''') # WARNING --- has been renamed and altered (renamed to configuration_representations and the primary key became autoincrement)               
    print("--- Table configuration_format created successfully")

    conn.execute('''DROP TABLE  IF EXISTS configuration_measurements;''') 
    conn.execute('''CREATE TABLE configuration_measurements
            (configuration_id INTEGER, 
            energy     FLOAT,
            workload       BIGINT,
            energy_efficiency FLOAT,
            FOREIGN KEY(configuration_id) REFERENCES configuration_representations(configuration_id));''') 
    print("--- Table configuration_measurements created successfully")

    conn.execute('''DROP TABLE IF EXISTS configuration_efficiency_estimation;''') 
    conn.execute('''CREATE TABLE configuration_efficiency_estimation
            (configuration_id INTEGER,
            train_or_test_set VARCHAR(50),
            energy_efficiency FLOAT,
            FOREIGN KEY(configuration_id) REFERENCES configuration_representations(configuration_id));''')
    print("--- Table configuration_measurements created successfully")
 
    conn.execute('''DROP TABLE IF EXISTS configuration_description__google_pixel_4a_5g;''') 
    conn.execute('''CREATE TABLE configuration_description__google_pixel_4a_5g
            (configuration_id INTEGER,
            little_socket_frequency INT,
            core_0_state BIT,
            core_1_state BIT,
            core_2_state BIT, 
            core_3_state BIT,
            core_4_state BIT,
            core_5_state BIT,
            core_6_state_freq_level TINYINT,
            core_7_state_freq_level TINYINT,
            FOREIGN KEY(configuration_id) REFERENCES configuration_representations(configuration_id));''')
    print("--- Table configuration_description__google_pixel_4a_5g created successfully")
    




def is_train_or_test(config_in_model_format, X_train, X_test ):
    result = ""
    if ( utils.array_in_list_of_array(config_in_model_format, X_train)) != -1: 
        result =  "train"
    elif ( utils.array_in_list_of_array(config_in_model_format, X_test)) != -1: 
        result = "test"
    return result


MAX_NUMBER = 10000000
def fill_database(conn, X_user_friendly, X, y, energy_array, workload_array, X_train, y_train, X_test, y_test, 
             gauss_process, experiment_format = "google_pixel_4a_5g_format", X_format_in_model = "base_Y", exact_values_of_frequency = False,
             table_name = "all"):
  
    
    
 
    """
    """
    print("getting estimation array") # the estismation of tested configurations will be add first in the configuration_efficiency_estimation table
    configuration_estimation_array  = get_and_write_sorted_estimations_of_configurations(MAX_NUMBER, gauss_process, experiment_format, X_format_in_model, exact_values_of_frequency)
    #[combination, combination_for_estimation, predicted_y_as_array[0]]
    #[list like [a,b,c,...], list like [Base Y format], float number  ]

    if table_name == "configuration_representations" or table_name == "all":
        print("--- Filling configuration_representations table")
        expe_format__model_format__efficiency_predicted = []
        configuration_representations_insertion_command=" INSERT INTO configuration_representations ( user_friendly_format, experiment_format, base_Y_format) VALUES"
        for expe_format__model_format__efficiency_predicted in configuration_estimation_array:
            x_user_friendly = convert_from_experiment_format_to_user_friendly (expe_format__model_format__efficiency_predicted[0], experiment_format)#X_user_friendly[config_index] 
            conguration_in_experiment_format_str =  str(expe_format__model_format__efficiency_predicted[0])  # str(convert_from_model_format_to_configuration(config_in_model_format))
            config_in_model_format_str = str(expe_format__model_format__efficiency_predicted[1])
            configuration_representations_insertion_command = configuration_representations_insertion_command + \
                    " \n ( \"" + x_user_friendly +  "\", \"" + conguration_in_experiment_format_str +  "\", \"" + config_in_model_format_str + "\" ),"
        configuration_representations_insertion_command = configuration_representations_insertion_command[:-1]+";"
        print(" configuration_representations isertion command :", configuration_representations_insertion_command)
        
        conn.execute(configuration_representations_insertion_command)
        conn.commit()
        print("--- configuration_representations table filled successfully")
        print("--- Printing  configuration_representations records")
        cursor = conn.execute("SELECT configuration_id, user_friendly_format, experiment_format, base_Y_format from configuration_representations")
        for row in cursor:
            print("---")
            print("configuration_id = ", row[0])
            print("user_friendly_format = ", row[1])
            print("experiment_format = ", row[2])
            print("base_Y_format = ", row[3],)
        print("--- Operation done successfully")
        
    if table_name == "configuration_description__google_pixel_4a_5g and configuration_efficiency_estimation"  or table_name == "all":
        print("filling configuration_description__google_pixel_4a_5g and configuration_efficiency_estimation table")
        configuration_description_insertion_command=" INSERT INTO configuration_description__google_pixel_4a_5g ( configuration_id, little_socket_frequency," + \
                                                "core_0_state, core_1_state, core_2_state, core_3_state," + \
                                                "core_4_state, core_5_state, core_6_state_freq_level, core_7_state_freq_level) VALUES" 

        configuration_efficiency_estimation_insertion_command=" INSERT INTO configuration_efficiency_estimation ( configuration_id, train_or_test_set, energy_efficiency ) VALUES"



        expe_format__model_format__efficiency_predicted = []
        for expe_format__model_format__efficiency_predicted in configuration_estimation_array:
            config_in_model_format_str = str(expe_format__model_format__efficiency_predicted[1])
            print("--- Getting the id of the configuration from configuration_representations table")
            cursor = conn.execute("SELECT configuration_id from configuration_representations WHERE base_Y_format  == '" + config_in_model_format_str + "';")
            configuration_id = int(cursor.fetchall()[0][0])
            print("configuration_id", configuration_id)

            model_repr = expe_format__model_format__efficiency_predicted[1]
            configuration_description_insertion_command = configuration_description_insertion_command + \
                    " \n ( " + str(configuration_id) +  ", " + str(model_repr[0]) +  "," + \
                    str(model_repr[1]) +  "," +  str(model_repr[2]) +  "," + str(model_repr[3]) +  "," + \
                    str(model_repr[4]) +  "," + str(model_repr[5]) +  "," + str(model_repr[6]) +  "," +  \
                    str(model_repr[7]) +  "," + str(model_repr[8]) +  "),"



            test_or_test = is_train_or_test( model_repr, X_train, X_test)
            efficiency_estimation = expe_format__model_format__efficiency_predicted[2]
            configuration_efficiency_estimation_insertion_command = configuration_efficiency_estimation_insertion_command + \
                    " \n ( " + str(configuration_id) +  ", \"" + test_or_test +  "\"," +  str(efficiency_estimation) + "),"

        
        configuration_description_insertion_command = configuration_description_insertion_command[:-1]+";"
        print("configuration_description insertion command :", configuration_description_insertion_command)

        configuration_efficiency_estimation_insertion_command = configuration_efficiency_estimation_insertion_command[:-1]+";"
        print("configuration_efficiency_estimation insertion command :", configuration_efficiency_estimation_insertion_command)

        
        conn.execute(configuration_description_insertion_command)
        conn.commit()
        print("--- configuration_description__google_pixel_4a_5g table filled successfully")
        print("--- Printing  configuration_description__google_pixel_4a_5g records")
        cursor = conn.execute("SELECT configuration_id, little_socket_frequency," + \
                                                "core_0_state, core_1_state, core_2_state, core_3_state," + \
                                                "core_4_state, core_5_state, core_6_state_freq_level, core_7_state_freq_level from configuration_description__google_pixel_4a_5g")
        for row in cursor:
            print("---")
            print("configuration_id = ", row[0])
            print("little_socket_frequency = ", row[1])
            print("core_0_state = ", row[2])
            print("core_6_state_freq_level = ", row[8],)
            print("core_7_state_freq_level = ", row[9],)
        print("--- Operation done successfully")
        
        conn.execute(configuration_efficiency_estimation_insertion_command)
        conn.commit()
        print("--- configuration_efficiency_estimation table filled successfully")
        print("--- Printing  configuration_efficiency_estimation records")
        cursor = conn.execute("SELECT configuration_id, train_or_test_set, energy_efficiency from configuration_efficiency_estimation")
        for row in cursor:
            print("---")
            print("configuration_id = ", row[0])
            print("train_or_test_set = ", row[1])
            print("energy_efficiency = ", row[2])
        print("--- Operation done successfully")
        
        

    if table_name == "configuration_measurements" or table_name == "all":
        print("filling configuration_measurements table")
        configuration_measurements_insertion_insertion_command = " INSERT INTO configuration_measurements ( configuration_id, energy, workload, energy_efficiency) VALUES" 

        config_index = 0
        config_in_model_format = []
        for config_in_model_format in X:
            config_in_model_format_str = str(config_in_model_format)
            print("--- Getting the id of the configuration from configuration_representations table, X = ", config_in_model_format)
            cursor = conn.execute("SELECT configuration_id from configuration_representations WHERE base_Y_format  == '" + config_in_model_format_str + "';")
            configuration_id = int(cursor.fetchall()[0][0])
            print("configuration_id", configuration_id)

            test_or_test = is_train_or_test( config_in_model_format, X_train, X_test)
            configuration_measurements_insertion_insertion_command = configuration_measurements_insertion_insertion_command + \
                    " \n ( " + str(configuration_id) +  ", " + str(energy_array[config_index]) + \
                    "," + str(int(workload_array[config_index])) +  ", " + str(y[config_index]) + "),"
            config_index = config_index + 1
        
        configuration_measurements_insertion_insertion_command = configuration_measurements_insertion_insertion_command[:-1]+";"
        print("configuration_measurements insertion command :", configuration_measurements_insertion_insertion_command)

        conn.execute(configuration_measurements_insertion_insertion_command)
        conn.commit()
        print("--- configuration_measurements table filled successfully")
        print("--- Printing  configuration_measurements records")
        cursor = conn.execute("SELECT configuration_id, energy, workload, energy_efficiency from configuration_measurements")
        for row in cursor:
            print("---")
            print("configuration_id = ", row[0])
            print("energy = ", row[1])
            print("worklad = ", row[2])
            print("energy efficiency = ", row[3],)
        print("--- Operation done successfully")
        
   

import lesson_learned_validation_code as validation_code
def validate_lesson_learned (conn, marginal_effect_exploration_folder_, output_file_name = "", output_plot_name = "",  paper_fontsize = 20, avg_marginal_score_table = []):
   
   
    print("--- Validating lesson learned:")

    ##Code to generate precision plot

    

    secondary_columns = ["little_socket_frequency",
            "core_0_state",
            "core_1_state",
            "core_2_state", 
            "core_3_state",
            "core_4_state",
            "core_5_state",
            "core_6_state_freq_level",
            "core_7_state_freq_level"]
    result_table = validation_code.validate_fitsAll_interaction_table(conn,"core_0_state",["0","1"], secondary_columns,  avg_marginal_score_table)
    
    print("Valitation table computed  ", result_table) 

    
 
    
    #validation_code.validate_fitsAll_advice(conn, "core_0_state",["0","1"], "little_socket_frequency", "0" , 1733811287.7068696)
    #validation_code.validate_fitsAll_increment(conn, "core_0_state", "0", "1", "little_socket_frequency", "0")

    """    
    if (output_file_name != ""):   
        output_file_path = marginal_effect_exploration_folder + "/" + output_file_name 
        with open(output_file_path,'w') as file:
            file.write("variable to increase, chipset state,  suitable-or-contraindicated, validation score, [accepted transition(s)], [rejected transition(s)]\n")
            file.write("\n") # normally when writing in the output csv file we should respect the same order present in this file [1]
                            # [1] https://docs.google.com/document/d/1A1u-KLEoS9BnJ_jWlHyuEksPFDvsR9Av021OCRyag3Q/edit
                            # but we did not fill the output csv file in this order
                            # it is why we add a void when we skipped lesson learned in the file pointed above. 
            file.write("core 0 state, core 0 ON or OFF,  suitable-the efficiency should increase, NOT_COMPUTABLE, NULL , NULL \n")
            file.write( validation_code.validate__scheduling_thread_on_core_0_no_matter_core_1_state(conn, "increases") + "\n")      
            file.write( validation_code.validate__scheduling_thread_on_little_core_i_no_matter_core_j_state(0, 2, conn, "increases") + "\n")      
            file.write( validation_code.validate__scheduling_thread_on_little_core_i_no_matter_core_j_state(0, 3, conn, "increases") + "\n")      
            file.write( validation_code.validate__scheduling_thread_on_little_core_i_no_matter_core_j_state(0, 4, conn, "increases") + "\n") 
            file.write( validation_code.validate__scheduling_thread_on_little_core_i_no_matter_core_j_state(0, 5, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_core_i_when_condition_on_socket_frequency(0, "medium", "low or medium",conn, "increases") + "\n")      

            file.write("\n")
            file.write( validation_code.validate__increasing_little_sockect_freq_when_core_6_state_freq_level_is_3(conn, "decreases") + "\n")    
            file.write("\n")
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(6, 0, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(6, 1, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(6, 2, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(6, 3, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(6, 4, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(6, 5, conn, "increases") + "\n")  

            file.write("\n")
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(7, 0, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(7, 1, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(7, 2, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(7, 3, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(7, 4, conn, "increases") + "\n")  
            file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(7, 5, conn, "increases") + "\n")  
            #file.write( validation_code.validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state(6, 0, conn, "increases") + "\n")  
    """
    
   


    return result_table 
import math
def compute_static_score (conn, output_folder,  paper_fontsize = 20, 
                                     dataset_size = 536, acceptance_degree = 171):
   
   
    print("--- Validating lesson learned:")
    """
    ax1 = fig.add_subplot(111)

    ax1.plot(range(5), range(5))

    ax1.grid(True)

    ax2 = ax1.twiny()
    ax2.set_xticks( ax1.get_xticks() )
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([x * 2 for x in ax1.get_xticks()])

    title = ax1.set_title("Upper x-axis ticks are lower x-axis ticks doubled!")
    title.set_y(1.1)
    fig.subplots_adjust(top=0.85)

    fig.savefig("1.png")
    """
    
    ##Code to generate precision plot

    output_file_name = "static_score_according_to_acceptance_degree_" + str(dataset_size) + ".png" 

    acceptance_degrees = np.arange(start=1, stop=dataset_size, step=10).tolist()
    #acceptance_ratio = [round(x / dataset_size,2) for x in acceptance_degrees]
    precision = [validation_code.compute_static_score(conn, a_d)  for a_d in acceptance_degrees ]
    
    paper_fontsize = 16
    fig = plt.figure()
    ax_degree = fig.add_subplot(111)
    ax_degree.plot(acceptance_degrees, precision ,  color='black', marker='o', linestyle='-' )
    ax_degree.set_xlabel( "Acceptance degree" ,  fontsize = paper_fontsize )
    ax_degree.set_ylabel( "FitsAll static score or precision", fontsize = paper_fontsize)
    label_format = '{:,.0f}'
    ax_degree.set_yticklabels( [label_format.format(x) for x in ax_degree.get_yticks().tolist()]  ,  fontsize = paper_fontsize)


################ workload subplot
## google pixel bars


    ax_ratio = ax_degree.twiny()
    ax_ratio.set_xticks(ax_degree.get_xticks())
    ax_ratio.set_xticklabels([round(x / dataset_size,2) for x in ax_degree.get_xticks()], fontsize = paper_fontsize)
    ax_ratio.set_xlabel( "Acceptance ratio ", fontsize = paper_fontsize)
   
    #static_score_plot = plt.figure()
    #plt.plot(acceptance_degrees, precision ,  color='black', marker='o', linestyle='-' )
    #plt.yticks(fontsize=paper_fontsize)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(paper_fontsize)
    plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()       
    plt.savefig(output_folder + "/"+ output_file_name )
    plt.clf()
    plt.cla()
    plt.close()

    print("plot produced in file ",output_folder + "/" + output_file_name)
    if (output_file_name != ""):   
        output_file_path = output_folder + "/" + output_file_name + ".csv" 
        with open(output_file_path,'w') as file:
            file.write("acceptance_degree, static score\n")
            acceptance_degree_precision_array = [ [ac_d, pr] for ac_d, pr in zip(acceptance_degrees, precision)]
            sorted_acceptance_degree_precision_array = sorted(acceptance_degree_precision_array, key=lambda kv: kv[1],  reverse=True) 
            for couple in sorted_acceptance_degree_precision_array:
                file.write(str(couple[0]) + ", "  + str(couple[1]) + "\n")
          
    
    static_score = validation_code.compute_static_score(conn, acceptance_degree)
    return  static_score


    

    """ OLD COMMANDS USED TO TEST IF THE DATABASE WAS FILLED CORRECTLY
    
    command = '''
        SELECT
            configuration_description_little_freq_1_Medium_socket_freq_H.configuration_id,
            configuration_description_little_freq_1_Medium_socket_freq_H.little_socket_frequency,
            configuration_description_little_freq_1_Medium_socket_freq_H.core_6_state_freq_level
            FROM
                (SELECT
                    configuration_description_Medium_socket_freq_H.configuration_id,
                    configuration_description_Medium_socket_freq_H.little_socket_frequency,
                    configuration_description_Medium_socket_freq_H.core_6_state_freq_level
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.little_socket_frequency,
                        configuration_description__google_pixel_4a_5g.core_6_state_freq_level
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_6_state_freq_level == 3) /* see if we can reduce this */  
                    AS configuration_description_Medium_socket_freq_H
                WHERE
                    configuration_description_Medium_socket_freq_H.little_socket_frequency == 1) 
                AS  configuration_description_little_freq_1_Medium_socket_freq_H
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id);
    '''

    command = '''
        SELECT
            configuration_description_little_freq_0_Medium_socket_freq_H.configuration_id,
            configuration_description_little_freq_0_Medium_socket_freq_H.little_socket_frequency,
            configuration_description_little_freq_0_Medium_socket_freq_H.core_6_state_freq_level
        FROM
                (SELECT
                    configuration_description_Medium_socket_freq_H.configuration_id,
                    configuration_description_Medium_socket_freq_H.little_socket_frequency,
                    configuration_description_Medium_socket_freq_H.core_6_state_freq_level
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.little_socket_frequency,
                        configuration_description__google_pixel_4a_5g.core_6_state_freq_level
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_6_state_freq_level == 3) /* see if we can reduce this */  
                    AS configuration_description_Medium_socket_freq_H
                WHERE
                    configuration_description_Medium_socket_freq_H.little_socket_frequency == 0) 
                AS  configuration_description_little_freq_0_Medium_socket_freq_H 
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id); '''
    """
    """
    command = '''
       
                SELECT
                    configuration_description_Medium_socket_freq_H.configuration_id,
                    configuration_description_Medium_socket_freq_H.little_socket_frequency,
                    configuration_description_Medium_socket_freq_H.core_6_state_freq_level
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.little_socket_frequency,
                        configuration_description__google_pixel_4a_5g.core_6_state_freq_level
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_6_state_freq_level == 3) /* see if we can reduce this */  
                    AS configuration_description_Medium_socket_freq_H
                WHERE
                    configuration_description_Medium_socket_freq_H.little_socket_frequency == 0
              ; '''
    """
    """
    command = '''SELECT * FROM configuration_representations;'''
    cursor = conn.execute(command)
    print("--- configuration_representations ")
    for row in cursor:
        print("row = ", row)

    command = '''SELECT * FROM configuration_description__google_pixel_4a_5g;'''
    cursor = conn.execute(command)
    print("--- configuration_description__google_pixel_4a_5g ")
    for row in cursor:
        print("row = ", row)
    
    command = '''SELECT * FROM configuration_representations;'''
    cursor = conn.execute(command)
    print("--- configuration_representations ")
    for row in cursor:
        print("row = ", row)
    
    command = '''SELECT * FROM configuration_efficiency_estimation;'''
    cursor = conn.execute(command)
    print("--- configuration_efficiency_estimation ")
    for row in cursor:
        print("row = ", row)
        

    command = '''SELECT * FROM configuration_measurements;'''
    cursor = conn.execute(command)
    print("--- configuration_measurements ")
    """
    












    
def alter_database():   
    print("--- Creating / opening data base")
    conn = sqlite3.connect('experiments_and_estimations_results.db')
    print("--- Opened database successfully")

    # the commented blocks are the codes used to modify the database after its creation. 
    # I don't delete it to conserve modification history
    """
    print ("--- Altering configuration_efficiency_estimation table")
    conn.execute('''ALTER TABLE configuration_efficiency_estimation
        ADD COLUMN test_or_train_data  VARCHAR(50)  ;''') #can be "test" or "train"

    print("--- Table configuration_efficiency_estimation  altered successfully")
    """
    
    """
    print ("--- dropping configuration_format table")
    conn.execute('''DROP TABLE configuration_format;''') #can be "test" or "train"
    print ("---  configuration_format table dropped")
    """


    """
    conn.execute('''CREATE TABLE configuration_format
            (configuration_id INTEGER PRIMARY KEY AUTOINCREMENT    NOT NULL,
            user_friendly_format       VARCHAR(50),
            experiment_format          VARCHAR(50),
            base_Y_format       VARCHAR(50)
            );''')
    
    print("--- Table configuration_format created successfully")   
    """
    """
    conn.execute(''' ALTER TABLE configuration_format
        RENAME TO configuration_representations;''')
    print("--- Table configuration_format renamed successfully")   
    """

    """
    print("--- Clearing configuration_representations ")   
    conn.execute(''' DELETE FROM configuration_representations; ''')
    print("--- Table configuration_representations cleared successfully")   
    """

    """
    print("--- Deleting colomumn energy and workload from configuration_efficiency_estimation table ")
    conn.execute('''DROP TABLE configuration_efficiency_estimation;''') 
    conn.execute('''CREATE TABLE configuration_efficiency_estimation
            (configuration_id INT PRIMARY KEY     NOT NULL,
             train_or_test_set VARCHAR(50),
            energy_efficiency FLOAT);''') 
    print("--- Colomumn energy and workload correctly deleted from configuration_efficiency_estimation")  

    """
    """
    print("--- Renaming the table and inversing order of colomumns energy and workload from configuration_measurement table ")
    conn.execute('''DROP TABLE configuration_measurement;''') 
    conn.execute('''CREATE TABLE configuration_measurements
            (configuration_id INT PRIMARY KEY     NOT NULL,
            energy     FLOAT,
            workload       BIGINT,
            energy_efficiency FLOAT );''') 
    print("--- Colomumn energy and workload correctly inverte from configuration_measurement table and table renamed")  
    """
    
    print("--- adding foreign keys to tables configuration_measurement configuration_efficiency_estimation and configuration_description__google_pixel_4a_5g ")
    conn.execute('''DROP TABLE  IF EXISTS configuration_measurements;''') 
    conn.execute('''CREATE TABLE configuration_measurements
            (configuration_id INTEGER, 
            energy     FLOAT,
            workload       BIGINT,
            energy_efficiency FLOAT,
            FOREIGN KEY(configuration_id) REFERENCES configuration_representations(configuration_id));''') 
    conn.execute('''DROP TABLE IF EXISTS configuration_efficiency_estimation;''') 
    conn.execute('''CREATE TABLE configuration_efficiency_estimation
            (configuration_id INTEGER,
            train_or_test_set VARCHAR(50),
            energy_efficiency FLOAT,
            FOREIGN KEY(configuration_id) REFERENCES configuration_representations(configuration_id));''') 
    conn.execute('''DROP TABLE IF EXISTS configuration_description__google_pixel_4a_5g;''') 
    conn.execute('''CREATE TABLE configuration_description__google_pixel_4a_5g
            (configuration_id INTEGER,
            little_socket_frequency INT,
            core_0_state BIT,
            core_1_state BIT,
            core_2_state BIT, 
            core_3_state BIT,
            core_4_state BIT,
            core_5_state BIT,
            core_6_state_freq_level TINYINT,
            core_7_state_freq_level TINYINT,
            FOREIGN KEY(configuration_id) REFERENCES configuration_representations(configuration_id));''')
    print("--- foreign keys added")  
    
    print("--- Dumping the data base")
    cursor = conn.execute("SELECT * FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())
    print("--- Data base dumped")

