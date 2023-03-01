
import numpy as np
import math 
import matplotlib.pyplot as plt 
# functions 
#################################

def setup_X_format_meaning_dictionnaries(phone_name):
    if phone_name == "google_pixel_4a_5g" : 

        base_Y__X_meaning_dictionnary = {"X_0" : "frequency level of Little Socket",
                                    "X_1" : "Core 0 state", 
                                    "X_2" : "Core 1 state", 
                                    "X_3" : "Core 2 state", 
                                    "X_4" : "Core 3 state", 
                                    "X_5" : "Core 4 state",
                                    "X_6" : "Core 5 state",  
                                    "X_7" : "Medium Socket or core 6 frequency",
                                    "X_8" : "Big Socket or core 7 frequency"} 


        base_Y_N_on_socket__X_meaning_dictionnary = {"X_0" : "frequency level of Little Socket",
                                    "X_1" : "Number of little cores active",  
                                    "X_2" : "frequency level of Medium Socket or core 6",
                                    "X_3" : "frequency level of Big Socket or core 7"} 
        
        base_Y_F__X_meaning_dictionnary = {"X_0" : "Little Socket frequency is freed",
                                    "X_1" : "frequency level of Little Socket",
                                    "X_2" : "Core 0 state", 
                                    "X_3" : "Core 1 state", 
                                    "X_4" : "Core 2 state", 
                                    "X_5" : "Core 3 state", 
                                    "X_6" : "Core 4 state",
                                    "X_7" : "Core 5 state",  
                                    "X_8" : "Medium Socket frequency is freed",
                                    "X_9" : "Medium Socket or core 6 frequency",
                                    "X_10" : "Big Socket frequency is freed",
                                    "X_11" : "Big Socket or core 7 frequency"} 


        base_Y_F_N_on_socket__X_meaning_dictionnary = {"X_0" : "Little Socket frequency is freed",
                                    "X_1" : "frequency level of Little Socket",
                                    "X_2" : "Number of little cores active",  
                                    "X_3" : "Medium Socket frequency is freed",
                                    "X_4" : "frequency level of Medium Socket or core 6",
                                    "X_5" : "Big Socket frequency is freed",
                                    "X_6" : "frequency level of Big Socket or core 7"} 

    elif phone_name == "samsung_galaxy_s8" :

        base_Y__X_meaning_dictionnary = {"X_0" : "frequency level of Little Socket",
                                    "X_1" : "Core 0 state", 
                                    "X_2" : "Core 1 state", 
                                    "X_3" : "Core 2 state", 
                                    "X_4" : "Core 3 state",
                                    "X_5" : "frequency level of Big Socket",
                                    "X_6" : "Core 4 state",  
                                    "X_7" : "Core 5 state", 
                                    "X_8" : "Core 6 state", 
                                    "X_9" : "Core 7 state"} 

        base_Y_N_on_socket__X_meaning_dictionnary = {"X_0" : "frequency level of Little Socket",
                                    "X_1" : "Number of little cores active",  
                                    "X_2" :"frequency level of Big Socket",
                                    "X_3" : "Number of Big cores active"} 

        base_Y_F__X_meaning_dictionnary = {"X_0" : "Little Socket frequency is freed",
                                    "X_1" : "frequency level of Little Socket",
                                    "X_2" : "Core 0 state", 
                                    "X_3" : "Core 1 state", 
                                    "X_4" : "Core 2 state", 
                                    "X_5" : "Core 3 state",
                                    "X_6" : "Big Socket frequency is freed",
                                    "X_7" : "frequency level of Big Socket",
                                    "X_8" : "Core 4 state",  
                                    "X_9" : "Core 5 state", 
                                    "X_10" : "Core 6 state", 
                                    "X_11" : "Core 7 state"} 

        base_Y_F_N_on_socket__X_meaning_dictionnary = {"X_0" : "Little Socket frequency is freed",
                                    "X_1" : "frequency level of Little Socket",
                                    "X_1" : "Number of little cores active",  
                                    "X_0" : "Big Socket frequency is freed",
                                    "X_2" :"frequency level of Big Socket",
                                    "X_3" : "Number of Big cores active"} 

    return base_Y__X_meaning_dictionnary, base_Y_N_on_socket__X_meaning_dictionnary, base_Y_F__X_meaning_dictionnary, base_Y_F_N_on_socket__X_meaning_dictionnary


def get_for_the_paper_X_format_meaning_dictionnaries(phone_name):
    if phone_name == "google_pixel_4a_5g" : 
        base_Y__X_meaning_dictionnary = {"X_0" : "Little socket freq. level",
                                "X_1" : "Core 0 state", 
                                "X_2" : "Core 1 state", 
                                "X_3" : "Core 2 state", 
                                "X_4" : "Core 3 state", 
                                "X_5" : "Core 4 state",
                                "X_6" : "Core 5 state",  
                                "X_7" : "Medium socket state & freq.",
                                "X_8" : "Big socket state & freq."} 
    return base_Y__X_meaning_dictionnary

def convert_from_configuration_to_base_Y(configuration,  format="google_pixel_4a_5g_format"): # or format can be"samsung_galaxy_s8_format"?
    # convert a configuration to base  Y 
    # return the result as a numpy array

    # getting the frequency level (the first not nul value in the six first numbers of configurations)
                                   # (other not nul values in the six first numbers of the configuration should be the same)
    print (" --- Converting " +  repr(configuration) + " in base Y array notation" )

    result = []

    if format == "google_pixel_4a_5g_format":
        result.append(0)
        frequency_level = 0
        for index in range(0, 6):
            if configuration[index] != 0 :
                frequency_level = int(configuration[index])
                result.append(1)
            else: 
                result.append(0)
            
        if frequency_level == 0:
            result[0] = 0
        elif frequency_level == 4:
             result[0] =  2  # level 3 and level 4 of frequency in the input of this function are the same. 
        else :
            result[0] =  frequency_level - 1  
                            
        for index in range(6, len(configuration)):
            if configuration[index] == 4:
                result.append(3)  # level 3 and level 4 of frequency in the input of this function are the same. 
            else:
                result.append( int(configuration[index]))
        print (" --- Result = ",  result)
    elif   format == "samsung_galaxy_s8_format":
        print("--- Processing little cores")
        result.append(0)
        frequency_level = 0
        for index in range(0, 4):
            if configuration[index] != 0 :
                frequency_level =  int(configuration[index])
                result.append(1)
            else: 
                result.append(0)
            
        if frequency_level == 0:
            result[0] = 0
        elif frequency_level == 4:
             result[0] =  2  # level 3 and level 4 of frequency in the input of this function are the same. 
        else :
            result[0] =  frequency_level - 1  
 
        print("--- Processing big cores")
        result.append(0)
        frequency_level = 0
        for index in range(4, 8):
            if  int(configuration[index]) != 0 :
                frequency_level =  int(configuration[index])
                result.append(1)
            else: 
                result.append(0)
            
        if frequency_level == 0:
            result[5] = 0
        elif frequency_level == 4:
            result[5] = 2  # level 3 and level 4 of frequency in the input of this function are the same. 
        else :
            result[5] = frequency_level - 1  
       

        print (" --- Result _ samsung = ",  result)
    else:
        return -1
    return result


def convert_from_configuration_to_base_Y_F(configuration,  format="google_pixel_4a_5g_format"): # or  can be"samsung_galaxy_s8_format"?
    # convert a configuration to base  Y with the possibility to have frequency freed (F)
    # return the result as a numpy array

    # getting the frequency level (the first not nul value in the six first numbers of configurations)
                                   # (other not nul values in the six first numbers of the configuration should be the same)
    print (" --- Converting " +  repr(configuration) + " in base Y F array notation" )

    result = []

    if format == "google_pixel_4a_5g_format":
        result.append(0) # for frequency freed (0 if not freed, 1 if freed)
        result.append(0) # for frequency level if not (from 0 to 2)
        frequency_level = 0
        for index in range(0, 6):
            if configuration[index] != 0:
                frequency_level =  int(configuration[index])
                result.append(1)
            else: 
                result.append(0)
            
        if frequency_level == 0:
            result[0] = 0
            result[1] = 0
        elif frequency_level == 4:
            result[0] =  1
            result[1] =  3  # we set the frequency level to 3 to not alter result on other possible value of X[1], 
                            # so that when X[1] is 0,1,2 it forcely means that frequency has been modify, 
                            # so when interpreting model result, the value of 3 of this variable can be neglected.  
        else:
            result[0] = 0
            result[1] =  frequency_level - 1  


        for index in range(6, len(configuration)):
            # we append the frequency_freed variable
            if configuration[index] == 4:
                result.append(1)
            else:
                result.append(0)
            # we append the frequency variable (if 4 we let it like that for the same reason as for the frequency level value 3 of the little socket)
            result.append( int(configuration[index]))

        print (" --- Result = ",  result)

    elif   format == "samsung_galaxy_s8_format":
        print("--- Processing little cores")
        result.append(0)
        result.append(0)
        frequency_level = 0
        for index in range(0, 4):
            if configuration[index] != 0 :
                frequency_level =  int(configuration[index])
                result.append(1)
            else: 
                result.append(0)
            
        if frequency_level == 0:
            result[0] = 0
            result[1] = 0
        elif frequency_level == 4:
            result[0] = 1
            result[1] = 3
        else:
            result[0] =  0
            result[0] =  frequency_level - 1  
 
        print("--- Processing big cores")
        result.append(0)
        result.append(0)
        frequency_level = 0
        for index in range(4, 8):
            if configuration[index] != 0:
                frequency_level =  int(configuration[index])
                result.append(1)
            else: 
                result.append(0)
            
        if frequency_level == 0:
            result[6] = 0
            result[7] = 0
        elif frequency_level == 4:
            result[6] = 1
            result[7] = 3
        else:
            result[6] = 0
            result[7] = frequency_level - 1  
             
        print (" --- Result _ samsung = ",  result)
    else:
        return -1
    return result



def convert_from_configuration_to_base_Y_N_on_socket(configuration, format="google_pixel_4a_5g_format"): # or  can be"samsung_galaxy_s8_format".
    # convert a configuration to base Y but with the number of cores on little socket not the core states on little sockets. 
    # return the result as a numpy array

    # getting the frequency level setted on the socket (the last not nul value in the six first numbers of the configurations)
                                   # (other not nul values in the six first numbers of the configuration should be the same)
    print (" --- Converting " +  repr(configuration) + " in -base Y, N on little- array notation" )

    result = []
    

    if format == "google_pixel_4a_5g_format":
        frequency_level = 0
        number_of_little_cores_active = 0 
        for index in range(0, 6):
            if configuration[index] != 0 :  
                frequency_level =  int(configuration[index])
                #result.append(1)
                number_of_little_cores_active = number_of_little_cores_active + 1

        if frequency_level == 0:
            result.append(0)
        elif frequency_level == 4:
            result.append(2)  # level 3 and level 4 of frequency in the input of this function are the same. 
        else :
            result.append(frequency_level - 1)

        result.append(number_of_little_cores_active) 

        for index in range(6, len(configuration)):
            if(configuration[index]==4):
                result.append(3)
            else:
                result.append( int(configuration[index]))
    
    elif format == "samsung_galaxy_s8_format":
        print("--- Processing little cores")
        frequency_level = 0
        number_of_little_cores_active = 0 
        for index in range(0, 4):
            if configuration[index] != 0 :  
                frequency_level = int(configuration[index])
                #result.append(1)
                number_of_little_cores_active = number_of_little_cores_active + 1

        if frequency_level == 0:
            result.append(0)
        elif frequency_level == 4:
            result.append(2)  # level 3 and level 4 of frequency in the input of this function are the same. 
        else :
            result.append(frequency_level - 1)


        result.append(number_of_little_cores_active) 

        print("--- Processing big cores")
        frequency_level = 0
        number_of_big_cores_active = 0 
        for index in range(4, 8):
            if configuration[index] != 0 :  
                frequency_level =  int(configuration[index])
                #result.append(1)
                number_of_big_cores_active = number_of_big_cores_active + 1

        if frequency_level == 0:
            result.append(0)
        elif frequency_level == 4:
            result.append(2)  # level 3 and level 4 of frequency in the input of this function are the same. 
        else :
            result.append(frequency_level - 1)

      

        result.append(number_of_big_cores_active) 
    else: 
        return -1

    print (" --- Result = ",  result)
    return result


def convert_from_configuration_to_base_Y_F_N_on_socket(configuration, format="google_pixel_4a_5g_format"):
    # convert a configuration to base Y N and at the same time adding variable reflexing the fact that frequency is freed or not
    print (" --- Converting " +  repr(configuration) + " in -base Y F, N on little- array notation" )

    result = []
    

    if format == "google_pixel_4a_5g_format":
        frequency_level = 0
        number_of_little_cores_active = 0 
        for index in range(0, 6):
            if configuration[index] != 0 :  
                frequency_level =  int(configuration[index])
                #result.append(1)
                number_of_little_cores_active = number_of_little_cores_active + 1

        if frequency_level == 0:
            result.append(0)
            result.append(0)
        elif frequency_level == 4:
            result.append(1)
            result.append(3)
        else:
            result.append(0)
            result.append(frequency_level - 1)

        result.append(number_of_little_cores_active) 

        for index in range(6, len(configuration)):
            # we append the frequency_freed variable
            if configuration[index] == 4:
                result.append(1)
            else:
                result.append(0)
            # we append the frequency variable (if 4 we let it like that for the same reason as for the frequency level value 3 of the little socket)
            result.append(int(configuration[index]))
    
    elif format == "samsung_galaxy_s8_format":
        print("--- Processing little cores")
        frequency_level = 0
        number_of_little_cores_active = 0 
        for index in range(0, 4):
            if configuration[index] != 0 :  
                frequency_level =  int(configuration[index])
                #result.append(1)
                number_of_little_cores_active = number_of_little_cores_active + 1

        if frequency_level == 0:
            result.append(0)
            result.append(0)
        elif frequency_level == 4:
            result.append(1)
            result.append(3)
        else:
            result.append(0)
            result.append(frequency_level - 1)


        result.append(number_of_little_cores_active) 


        print("--- Processing big cores")
        frequency_level = 0
        number_of_big_cores_active = 0 
        for index in range(4, 8):
            if configuration[index] != 0 :  
                frequency_level =  int(configuration[index])
                #result.append(1)
                number_of_big_cores_active = number_of_big_cores_active + 1

        if frequency_level == 0:
            result.append(0)
            result.append(0)
        elif frequency_level == 4:
            result.append(1)
            result.append(3)
        else:
            result.append(0)
            result.append(frequency_level - 1)

        result.append(number_of_big_cores_active) 
    else: 
        return -1

    print (" --- Result = ",  result)
    return result


def read_configuration_X(file_path):
    print("---> Reading the data set file X in generic format") 
    X = []
    with open(file_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            #print("reading line {}: {}".format(cnt, line.strip()))
            str_X_line = line.split(',')
            #print ("arrays as string : ", str_X_line)
            X_line = [float(numeric_string) for numeric_string in str_X_line]
            #print("resulted array : ", X_line)
            X.append(X_line)
            #print("after adding it to X : ", X)
            line = fp.readline()
            cnt += 1
    return X

def read_ratio_energy_by_workload(file_path):
    y = []
    print("---> Reading the data set y = ratio_energy_by_workload") 
    with open(file_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            #print("reading line {}: {}".format(cnt, line.strip()))
            y_line_numeric = float(line)
            #print("resulted y value : ", y_line_numeric)
            y.append(y_line_numeric)
            #print("after adding it to y : ", y_line_numeric)
            line = fp.readline()
            cnt += 1
    return y

def read_configuration_in_user_frendly_format(file_path):
    X = []
    print("---> Reading the data set file X in generic format") 
    with open(file_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            print("reading line {}: {}".format(cnt, line.strip()))
            line =line.strip()
            X.append(line)
            print("after adding it to X : ", X)
            line = fp.readline()
            cnt += 1
    return X

def create_dictionnary_for_X(X_user_friendly, X):
    Dict = {}
    print("---> Creating X dictionnary, from userfriendly values to x values") 
    index = 0
    for key in X_user_friendly:
        #print("creating value for key ", key)
        x_value = X[index]
        Dict[key] = x_value
        index +=1 
    return Dict

def get_x_from_x_user_value(x, X_dict):
    x_user_friendly = ""
    keys = X_dict.keys()
    for k in keys:
        #print("---> k: ", repr(k)) 
        #print("---> X_dict[k]: ", X_dict[k]) 
        #print("---> x: ", repr(x)) 
        if(X_dict[k] == x):
            x_user_friendly = k
    return x_user_friendly
   

def get_X_user_friendly_from_X(X, X_dict):
    X_user_friendly = []
    print("---> getting userfriendly values from X values") 
    for x in X:
        #print("getting user friendly format of :", x) 
        x_user_friendly = get_x_from_x_user_value(x,X_dict)
        #print("result:", x_user_friendly)
        X_user_friendly.append(x_user_friendly)
        
    
    return X_user_friendly


def sparce_X_to_get_impact_of_frequency(X):
    result = []
    for X_i in X:
        result_i = []
        for n in X_i:
            for k in range(0,n):
                result_i.append(1)
            for k in range(n,3):
                result_i.append(0)
        result.append(result_i)
    return result




### NOrmalizing X_train. 
def get_X_means(X): 
    M = len(X[1])
    print(" --- Start getting X_means, M = ", M)
    print(" ---  X = ", X)
    result = []
    for j in range(0,M):
        result.append(np.mean(X[:,j]))
    print(" --- Start getting X_means, result  = ", result)
    return result
def my_square(a):
    return a*a
v_square = np.vectorize(my_square)

def get_X_standard_deviation(X):
    X_means_as_row = get_X_means(X)
    M = len(X[1])
    N = len(X)
    print(" --- Start getting X standard deviation, M = ", M)
    result = []
    
    for j in range(0,M):
        print(" --- processing column "+ str(j) + ":  ", X[:,j])
        print("column mean vector= ", [X_means_as_row[j]] * N)
        temp_column_j_of_difference = np.subtract(X[:,j], [X_means_as_row[j]] * N)
        print(" diff vector ", temp_column_j_of_difference)
        temp_column_j_of_squares = v_square(temp_column_j_of_difference)
        print(" squared vector ", temp_column_j_of_squares)
        temp_standard_deviation = math.sqrt (np.sum(temp_column_j_of_squares) / N)
        result.append(temp_standard_deviation)
    
    print(" --- End getting X standard deviation, result = ", result)
    return result

def standartization_function (x, mean, standard_deviation):
    x = (x - mean)/standard_deviation
    return x

def standartize(X):
    print(" --- Start standartizing X")
    N = len(X)
    M = len(X[1])
    X_means_as_row = get_X_means(X)
    X_standart_deviation = get_X_standard_deviation(X)
    result = []
    for j in range(0,M):
        print(" --- processing standartization on column "+ str(j) + " : ", X[:,j])
        print("mean = ", X_means_as_row[j])
        print("standart_deviation = ",X_standart_deviation[j] )
        if X_standart_deviation[j] == 0:
            result.append(X[:,j])
            print("result on column " + str(j) +":", X[:,j])
            continue
        vectorized_standartization_function = np.vectorize(standartization_function)
        temp_result = vectorized_standartization_function(X[:,j], [X_means_as_row[j]] * N, [X_standart_deviation[j]]* N)        
        print("result on column " + str(j) +":", temp_result)
        result.append(temp_result)


    final_result = np.asarray(result).T
    print(" --- End standartizing X, result  = ", final_result)
    return final_result



### getting the labda ideal value


def compute_r2_score(y_test, y_predicted):
    # r2 = 1 - sum_on_y_length ( (y_test - y_predicted)2 )  / sum ( (ytest - ytest_mean)2 )
    print("Start computin r squared, result = ") 
    y_mean = np.mean(y_test)
    N = len(y_test)
    print("column mean vector= ", [y_mean] * N)
    temp_vector_of_difference_with_mean = np.subtract(y_test, [y_mean] * N)
    print(" diff with mean vector ", temp_vector_of_difference_with_mean)
    temp_vector_of_difference_with_mean_squared = v_square(temp_vector_of_difference_with_mean)
    print(" diff with mean vector squared  ", temp_vector_of_difference_with_mean_squared)

    temp_vector_of_difference_with_predicted = np.subtract(y_test, y_predicted)
    print(" diff with predicted vector ", temp_vector_of_difference_with_predicted)
    temp_vector_of_difference_with_predicted_squared= v_square(temp_vector_of_difference_with_predicted)
    print(" diff with predicted vector squared", temp_vector_of_difference_with_predicted_squared)

    result = 1 - np.sum(temp_vector_of_difference_with_predicted_squared)/np.sum(temp_vector_of_difference_with_mean_squared)
    print("End computing r squared, result = ", result)
    return result


############# using statsmodels API
import statsmodels
import statsmodels.api as sm
import statsmodels.sandbox as sm_sandbox
from statsmodels.sandbox.regression.kernridgeregress_class import GaussProcess
import numpy as np
from sklearn.model_selection import LeaveOneOut

def get_leave_one_out_error(X, y, sigma2 , current_lambda):
    loo = LeaveOneOut()
    error_vector = []
    print(" getting loo error of with lamda = " + str(current_lambda) + ", printing X, y") 
    print(X,y)
    for train_index, test_index in loo.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_, X_test_ = X[train_index], X[test_index]
        y_train_, y_test_ = y[train_index], y[test_index]
        #print("printing X_train, X_test")
        #print(X_train_, X_test_)
        #print("y_train, y_test")
        #print(y_train_, y_test_)
        
        gp1 = GaussProcess(X_train_, y_train_, #kernel=kernel_euclid,
                    scale = sigma2, ridgecoeff = current_lambda)

        gp1.fit(y_train_)
        y_predicted_ = gp1.predict(np.asarray(X_test_))  # I don't know why the predicte y for their online example gives me a n*1 matrix while for me it is a simple vector. 
        print(" y tested = ", y_test_[0])
        print("y  predicted = ", y_predicted_[0])
        print("error ", (y_predicted_[0] - y_test_[0])**2 )
        error_vector.append( (y_predicted_[0] - y_test_[0])**2 )
    result = np.sum(error_vector)/len(error_vector)
    print("error squared vector ",error_vector )
    print("Total loo_error ", result )
    return result 



def find_regularization_parameter(X,y, sigma2, search_strategy = "sequential",  ltolerance = 0.001, 
       lambda_min = 0.000000001, lambda_max = 100, max_iterations = 100, sequential_gap = 0.0001, dichotomic_or_all_values_exploration_progression_ratio = 100 ):   
    ## we suppose that the fonction that compute the leave one out loss has a unique variation pattern on interval [lambda_min, lamda_max] 
    print(" Start, findind regularisation parameter ")
    result = lambda_max
    error = ltolerance
    iteration = 0
    # first way to search lambda, by dichotomic search
    if search_strategy == "dichotomic" : 
        lambda_min_loo_error = get_leave_one_out_error(X, y, sigma2, lambda_min)
        lamda_max_loo_error = get_leave_one_out_error(X, y, sigma2, lambda_max)
        while ( abs(lamda_max_loo_error - lambda_min_loo_error)> ltolerance and iteration < max_iterations ):
            print("iteration " + str(iteration) + "current difference of  loo_error ", lamda_max_loo_error- lambda_min_loo_error )
            if(lambda_min_loo_error < lamda_max_loo_error):
                lambda_max = lambda_max - (lambda_max-lambda_min)/dichotomic_or_all_values_exploration_progression_ratio
                lamda_max_loo_error = get_leave_one_out_error(X, y, sigma2, lambda_max)
                result = lambda_max
                error = lamda_max_loo_error
            else:
                lambda_min = lambda_min + (lambda_max-lambda_min)/dichotomic_or_all_values_exploration_progression_ratio
                lambda_min_loo_error = get_leave_one_out_error(X, y, sigma2, lambda_min)
                result = lambda_min
                error  = lambda_min_loo_error
            iteration = iteration +1
        print ("End finding  regularisation parameter: number of iterations = "+ str(iteration)+", result = "+ str(result)+"  error = " + repr(error)
                 + " Last error difference:" +  repr(abs(lamda_max_loo_error- lambda_min_loo_error)))

    elif search_strategy == "sequential":
        # second  way to search lambda, by sequential  search with parameter gap moving a windows 
        lambda_min_loo_error = get_leave_one_out_error(X, y,sigma2, lambda_min)
        lambda_max = lambda_min + sequential_gap
        lamda_max_loo_error = get_leave_one_out_error(X, y,sigma2, lambda_max)
        iteration = 0 
        while ( abs(lamda_max_loo_error- lambda_min_loo_error)>ltolerance and iteration < max_iterations ):
            print("iteration " + str(iteration) + "current difference of  loo_error ", lamda_max_loo_error- lambda_min_loo_error )
            lambda_min = lambda_max 
            lambda_min_loo_error = lamda_max_loo_error

            lambda_max = lambda_max + sequential_gap
            lamda_max_loo_error = get_leave_one_out_error(X, y,sigma2, lambda_max)

            
            result = lambda_max
            error  = lamda_max_loo_error
            iteration = iteration +1
        print ("End finding  regularisation parameter: number of iterations = "+ str(iteration)+", result = "+ str(result)+"  error = " + repr(error)
                 + " Last error difference:" +  repr(abs(lamda_max_loo_error- lambda_min_loo_error)))
    
    elif search_strategy == "explore_all_values":
    # third strategy computing the leave one out error of all values of labda from labda min to labda max,   with step = (labda_max - labda_min)/ ratio. 
    # taking the the lambda with minimum l_o_o_error
        function_gap = (lambda_max - lambda_min)/dichotomic_or_all_values_exploration_progression_ratio
        min_loo_error  =  get_leave_one_out_error(X, y,sigma2, lambda_min)
        tmp_lambda = lambda_min + function_gap
        lambda___error_dict = {}
        lambda___error_dict[lambda_min] = min_loo_error
        while (tmp_lambda < lambda_max):
            
            tmp_loo_error = get_leave_one_out_error(X, y,sigma2, tmp_lambda)
            print("Testing lamda value : " + str(tmp_lambda) + " actual loo error: " +  str(tmp_loo_error) + ", min loo error : " + str(min_loo_error))
            if tmp_loo_error < min_loo_error :
                print(" One possible candidate of lamda : " + str(tmp_lambda) + ", loo error ", tmp_loo_error   ) 
                min_loo_error = tmp_loo_error
                result = tmp_lambda
                iteration = iteration + 1
                lambda___error_dict[tmp_lambda] = tmp_loo_error
            tmp_lambda = tmp_lambda + function_gap
        error = min_loo_error 
        sorted_lambda___error_dict = sorted(lambda___error_dict.items(), key=lambda kv: kv[1],  reverse=True)
        with open("loo_errors_according_to_lamda.csv",'w') as file:
            counter_value = 0
            for key, val in sorted_lambda___error_dict:
                print(" writing lamda value  ", key)
                print(" writing error ", val)    
                file.write(str(key) + "," + str(val))
                if (counter_value < len(sorted_lambda___error_dict)):
                    file.write('\n')
                    counter_value = counter_value + 1
        print ("End finding  regularisation parameter (with explore_all_values strategy): number of iterations = "+ str(iteration)+", result = "+ str(result)+"  error = " + repr(error))
        print("lamda value and loo error ", sorted_lambda___error_dict )
    return result, iteration, error   
  
    """
    # second  way to search lambda, by sequential  search with parameter gap, comaring loo_error with the previous value computed
    lamda___previous_loo_error = get_leave_one_out_error(X, y, lambda_max)
    lambda_min_loo_error = get_leave_one_out_error(X, y, lambda_min)
    iteration = 0 
    while ( abs(lambda_min_loo_error- lamda___previous_loo_error)>ltolerance and iteration < max_iterations ):
        print("iteration " + str(iteration) + "current difference of  loo_error ", lamda___previous_loo_error- lambda_min_loo_error )
        lamda___previous_loo_error = lambda_min_loo_error
        
        lambda_min = lambda_min + gap
        lambda_min_loo_error = get_leave_one_out_error(X, y, lambda_min)
        
        result = lambda_min
        error  = lambda_min_loo_error
        iteration = iteration +1
    print ("End finding  regularisation parameter: number of iterations = "+ str(iteration)+", result = "+ str(result)+"  error = ", error)
    """
    
    
   

import itertools
def array_in_list_of_array(val, list_of_array):
    place = 0
    for curr in list_of_array:
        if (np.array_equal(curr, val)):
            return place
        place = place + 1

    return -1

def positions_of_array_in_list_of_array(val, list_of_array):
    place = 0
    places = []
    for curr in list_of_array:
        if (np.array_equal(curr, val)):
            places.append(place)
        place = place + 1
    if(len(places) == 0):
        return [-1]
    return places

def compute_all_combination_of_thread_in_this_socket (X_socket, nmax = 1000):
    #this function takes as input a combination present on a socket, 
    # thread positions and frequencies and return all possible combinations of the threads at same frequencies on the same socket
    #ie (f1,f2,0,f2) will return it self plus
    #   (f1,f2,f2,0)
    #   (f1,0,f2,f2)
    #   (f2,f1,0,f2)
    #   (f2,f1,f2,0)
    #   (f2,0,f1,f2) 
    #   (f2,0,f2,f1)
    #   (f2,f2,0,f1)
    #   (f2,f2,f1,0)
    #   (0,f1,f2,f2)
    #   (0,f2,f1,f2)
    #   (0,f2,f2,f1)  
    #   if we have 4 cores on the socket, and 3 different values on each socket 
    #                            if  the first value appears a1 times, the second a2 times and the third a3 times
    #                              we should have 4! / (a1!*a2!*a3!)  possibles configurations 
    #                              in this example the number of configurations is 4*3*2*1/2 = 12 configurations

    # To compute it we first suppose that we have distinct values by making a link between the input combination and (0,1,2,3)
    # secondly we make all possible combination and remove duplicates in the antecedent set of the link. 
    ### Step 1: building relationships (dictionnary) between X = (f1,f2,0,f2) and X_dict_key = (0,1,2,3) (inversily in implementation)
    print("populating X", X_socket)
    first_dict_key = range(0,len(X_socket))
    dict = {}
    #dict [first_dict_key] = X_socket # optionnal
    all_key_permutations = list(itertools.permutations(first_dict_key))

    for current_permutation in all_key_permutations:
        current_X_socket = []
        for core in current_permutation : 
            current_X_socket.append(X_socket[core])
        dict[current_permutation] = current_X_socket

    print("dictionnary of possible permutations", dict)
    # Step 2 removing repetitions
    final_dict={}
    final_result = []
    number_of_combination_added = 0
    for key, val in dict.items():
        print(" processing key  ", key)
        print(" retaining this value ? ", val)
        print(" retained configurations ", final_result)
        if  array_in_list_of_array(val, final_result) == -1:
            final_result.append(val)
            final_dict[key] = val
            print(" answer : yes, not yet present")
            number_of_combination_added = number_of_combination_added + 1 
            if number_of_combination_added >= nmax:
                print(" we reached " + str(nmax) + " combinaisons for this configuration")
                break
        print(" answer : no, already present")
    print ("dictionnary of result", final_dict)
    print (" Number of combinaisons after populating ", number_of_combination_added)
    return final_result

def merge_configuration( populated_X_train,current_X_socket_combinations):
    for confg  in current_X_socket_combinations:
        populated_X_train.append(confg)
    return populated_X_train

def  get_socket_vectors(current_configuration, input_format):
    if input_format == "generic":
        little_socket_vector =  current_configuration[0:6]
        medium_socket_vector =  current_configuration[6:9]
        big_socket_vector = current_configuration[9:13]
        return little_socket_vector, medium_socket_vector, big_socket_vector
    elif input_format == "google_pixel_4a_5g":
        print("splitting configuration regarding to format :", input_format)
        little_socket_vector =  current_configuration[0:6]
        medium_socket_vector =  current_configuration[6:7]
        big_socket_vector = current_configuration[7:8]
        return little_socket_vector, medium_socket_vector, big_socket_vector
    #...

def populate_input_datas(X_train, y_train, input_format = "generic", nmax = 1000): #nmax is the maximum number of combinaison a single combinaison can produce 
    # augment the number of inputs data X_train and y_train by considering that 
    # the combinaison of the same number of thread on the same socket has the same 
    # energy efficiency.
    # for each different combination and y output: 
    #   split in three vectors litte socket, medium socket, big socket
    #   for each vector compute set of possible combinations (we obtain three sets)
    #   combines all combinations of three set (we obtain a list of possible X values )
    #   populate the y list with the current y value

    print ("populating X  values : ", X_train)
    print(" y value = ", y_train)

    populated_X_train = []
    populated_y_train = []
    for (current_configuration, y) in zip(X_train, y_train):
        print ("populating configuration : ", current_configuration)
        print(" y value = ", y)
        print("input format = ", input_format)
        little_socket_vector, medium_socket_vector, big_socket_vector = get_socket_vectors(current_configuration, input_format)
        conf_set_on_little_socket = compute_all_combination_of_thread_in_this_socket(little_socket_vector, nmax)
        conf_set_on_medium_socket = compute_all_combination_of_thread_in_this_socket(medium_socket_vector, nmax)
        conf_set_on_big_socket = compute_all_combination_of_thread_in_this_socket(big_socket_vector, nmax)
        current_X_socket_combinations = []
        number_of_combination = 0
        for little_confg in conf_set_on_little_socket:
            for medium_confg in conf_set_on_medium_socket:
                for big_confg in conf_set_on_big_socket:
                    print("Adding this configuration to result: ", np.concatenate((little_confg,medium_confg,big_confg)))
                    current_X_socket_combinations.append(np.concatenate((little_confg,medium_confg,big_confg)))
                    number_of_combination = number_of_combination +1 
        print("array of computed combinations ",current_X_socket_combinations)
        print("array of populated X before merging ", populated_X_train)
        print("array of populated y before merging ", populated_y_train)
        populated_X_train = merge_configuration( populated_X_train,current_X_socket_combinations)
        populated_y_train = np.concatenate((populated_y_train, [y] * number_of_combination ))
        print("array of populated X after merging ", populated_X_train)
        print("array of populated y after merging ", populated_y_train)

    print ("final populated X  values : ", populated_X_train)
    print(" final populated  y value = ", y)
    return populated_X_train, populated_y_train




def print_matrix_in_file(data,file_path, open_mode = 'w'):
    with open(file_path, open_mode) as file:
        line = []
        for line in data:
            counter_value = 1
            print(" --- Actual line:", line)
            for value in line:
                file.write(str(value))
                if (counter_value < len(line)):
                    file.write(',')
                    counter_value = counter_value + 1
            file.write('\n')

def capture_X_y_in_file(X, y, file_path):
    data_to_print = X.T 
    #print("Data after transposition : \n" ,data_to_print)

    data_to_print = np.append(data_to_print, [y], axis=0)
    
    
    
    #print("Data after adding y : \n" ,data_to_print)
    data_to_print = data_to_print.T
    #print("Data after retransposition: \n" ,data_to_print)
    header = []
    for i in range(0, len(X[0])):
        header.append("X_"+ str(i))
    header.append ("y")
    header = np.array(header)
    final_data_to_print = [header, *data_to_print]
    #print("Data to print after adding header : \n" ,final_data_to_print)
    print_matrix_in_file(final_data_to_print, file_path)

def is_from_manual_experiment(human_readable_configuration):
    """
    if(len(human_readable_configuration)==10)
        return False
    else
        return True
    """
    if(("m" in human_readable_configuration) or ("h" in human_readable_configuration) or (len(human_readable_configuration) < 8)):
        return True
    else:
        return False


def remove_duplicates(X_user_friendly, X,  y, energy_array = [], workload_array = [], value_to_retain = "median"):  
    final_X_user_friendly = []
    final_X = []
    final_y = []

    registered_duplicates = []
    index_of_loop = 0
    for val in X:
        print(" --- Checking value ", val)
        print(" --- Retained configurations ", final_X)
        place = array_in_list_of_array(val, final_X)
        if  place == -1 :
            final_X_user_friendly.append(X_user_friendly[index_of_loop])
            final_X.append(val)
            final_y.append(y[index_of_loop])
            print(" --- Answer : we add the configuration, it is  not yet present")  
        else :# (is_from_manual_experiment(X_user_friendly[index_of_loop])):
            print(" --- Answer : configuration is present, have it been processed? ", array_in_list_of_array(val, registered_duplicates))  
            if (array_in_list_of_array(val, registered_duplicates) == -1) :
                places = positions_of_array_in_list_of_array(val, X)
                print(" --- Answer : the configuration " + X_user_friendly[index_of_loop] +" is present in X at positions " + repr(places) )
                position_energy_array = []
                position_workload_array = []
                for duplicate_index in places:
                    print(" --- Position: " , duplicate_index )
                    print("--------------")
                    print(" --- Configuration: " , X_user_friendly[duplicate_index] )
                    print(" --- Energy efficiency: ", y[duplicate_index] )
                    print(" --- Energy: " , energy_array[duplicate_index] )
                    position_energy_array.append([duplicate_index, energy_array[duplicate_index]])
                    position_workload_array.append([duplicate_index, workload_array[duplicate_index]])

                    print(" --- Workload: ", workload_array[duplicate_index] )
                    print("--------------")

                # Now getting the median of duplicates, regarding the energy
                if value_to_retain == "median":
                    sorted_position_energy_array = sorted(position_energy_array, key=lambda kv: kv[1],  reverse=True) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
                    print("----------------------")
                    print("--- Ordered by energy, Printing the list of the " + str(len(places)) + " duplicates of " + repr(X_user_friendly[index_of_loop]))
                    for index in range(0,len(places)):                                     
                        position_in_data_point = sorted_position_energy_array[index][0]  
                        print("---  Duplicate  " + str(index) + " in the list of duplicate, And at position " + repr(position_in_data_point) + " in the X datas point")   
                        print("--------------")
                        print(" --- Configuration: " , X_user_friendly[position_in_data_point] )
                        print(" --- Energy efficiency: ", y[position_in_data_point] )
                        print(" --- Energy: " , energy_array[position_in_data_point] )
                        print(" --- Workload: ", workload_array[position_in_data_point] )
                        print("--------------")
                    median_index_in_the_duplicate_list = int(len(places)/2) - 1
                    median_couple_in_the_duplicate_list = sorted_position_energy_array[median_index_in_the_duplicate_list]       #  Can obtain something like (4, dist_3)
                    median_position_in_data_point = median_couple_in_the_duplicate_list[0]
                    print("--------------")
                    print("--- We append this Median as duplicate representant at position " + repr(median_index_in_the_duplicate_list) + " in the list of duplicates, And at position " + repr(median_position_in_data_point) + " in the X datas point")   
                    print("--------------")
                    print(" --- Configuration: ", X_user_friendly[median_position_in_data_point] )
                    print(" --- Energy efficiency: ", y[median_position_in_data_point] )
                    print(" --- Energy: " , energy_array[median_position_in_data_point] )
                    print(" --- Workload: ", workload_array[median_position_in_data_point] )
                    print("--------------")
                    final_X_user_friendly[place] = X_user_friendly[median_position_in_data_point]
                    final_X[place] = val
                    final_y[place] = y[median_position_in_data_point]
                elif value_to_retain == "mean":
                    sorted_position_energy_array = sorted(position_energy_array, key=lambda kv: kv[1],  reverse=True) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
                    print("---------------------- Listing and computing the mean")
                    print("--- Ordered by energy, Printing the list of the " + str(len(places)) + " duplicates of " + repr(X_user_friendly[index_of_loop]))
                    mean_energy = 0
                    mean_workload = 0
                    mean_efficiencty = 0
                    for index in range(0,len(places)):                                     
                        position_in_data_point = sorted_position_energy_array[index][0]  
                        print("---  Duplicate  " + str(index) + " in the list of duplicate, And at position " + repr(position_in_data_point) + " in the X datas point")   
                        print("--------------")
                        print(" --- Configuration: " , X_user_friendly[position_in_data_point] )
                        print(" --- Energy efficiency: ", y[position_in_data_point] )
                        print(" --- Energy: " , energy_array[position_in_data_point] )
                        print(" --- Workload: ", workload_array[position_in_data_point] )
                        print("--------------")
                        mean_energy = mean_energy +  energy_array[position_in_data_point]
                        mean_workload = mean_workload + workload_array[position_in_data_point]
                        mean_efficiencty = mean_efficiencty +  y[position_in_data_point]
                    mean_energy = mean_energy / len(places)
                    mean_workload = mean_workload / len(places)
                    mean_efficiencty = mean_efficiencty / len(places)
                    
                    print("--------------")
                    print("--- We append this mean as duplicate reprensentant in the X datas point")   
                    print("--------------")
                    print(" --- Configuration: ", X_user_friendly[position_in_data_point] )
                    print(" --- Energy efficiency: ", mean_efficiencty )
                    print(" --- Energy: " , mean_energy )
                    print(" --- Workload: ", mean_workload )
                    print("--------------")
                    final_X_user_friendly[place] = X_user_friendly[position_in_data_point]
                    final_X[place] = val
                    final_y[place] = mean_efficiencty

                registered_duplicates.append(val)
        index_of_loop = index_of_loop +1
    
    print ("final_X_user friendly : \n ", final_X_user_friendly)
    print ("final_X : \n ", final_X)
    print ("final_y : \n ", final_y)
    return final_X_user_friendly, final_X, final_y


def write_result_in_csv_file(file_path, data):
    with open(file_path,'a') as file:
        counter_value = 0
        print(" --- Actual line:", data)
        for value in data:
            file.write(str(value))
            if (counter_value < len(data)):
                file.write(',')
                counter_value = counter_value + 1
        file.write('\n')


def capture_kernel_means_marginal_and_linear_model_coeff(margins, 
       linear_coefficients, linear_coeff_vs_kernel_ridge_margins_file, X_meaning_dictionnary):
    
    X_header = []
    X_variable_meanings = []
    for i in range(0, len(linear_coefficients)):
        X_header.append("X_"+ str(i))
        X_variable_meanings.append(X_meaning_dictionnary["X_"+ str(i)])
    X_header_and_numbers = []
    X_header_and_numbers.append(X_header)
    X_header_and_numbers.append(X_variable_meanings)
    X_header_and_numbers.append(margins)
    X_header_and_numbers.append(linear_coefficients)
    X_header_and_numbers.append(abs(np.subtract(linear_coefficients, margins)))
    print(" X header, margins and linear coef numbers before transpose: \n", X_header_and_numbers)
    X_header_and_numbers_T = np.array(X_header_and_numbers).T
    X_header_and_numbers_T = X_header_and_numbers_T.tolist()
    for j in range(2,5):
        for i in range(0,len(margins)):
            print (" --- Before Modifying variable " +  X_header_and_numbers_T[i][j] + ", Type : " + str(type( X_header_and_numbers_T[i][j])))
            X_header_and_numbers_T[i][j] = float(X_header_and_numbers_T[i][j])
            print (" --- After Modifying variable " + str( X_header_and_numbers_T[i][j])+ ", Type : " + str(type( X_header_and_numbers_T[i][j])))


    
    print(" X header, margins and linear coef numbers after transpose: \n", X_header_and_numbers_T)
    summary_header = np.array(["X_variable", "meaning ", "kernel ridge margins", "linear regression coefficients", "difference"])
    summary_to_print = [summary_header, *X_header_and_numbers_T]
    print("margins and linearcoef summary_to_print : \n" ,summary_to_print)
    print_matrix_in_file(summary_to_print, linear_coeff_vs_kernel_ridge_margins_file, 'w')

    with open(linear_coeff_vs_kernel_ridge_margins_file,'a') as file:
        file.write('\n\n Ordered by kernel ridge coefficients, higher is better \n ')
    X_header_and_numbers_T_order_by_margin =  sorted(X_header_and_numbers_T, key=lambda kv: kv[2],  reverse=True) 
    print(" X header, margins and linear coef numbers after transpose, ordered by margin: \n", X_header_and_numbers_T_order_by_margin)
    summary_header = np.array(["X_variable",  "meaning ", "kernel ridge margins", "linear regression coefficients", "difference"])
    summary_to_print_ordered_by_margin = [summary_header, *X_header_and_numbers_T_order_by_margin]
    print("margins and linearcoef summary_to_print ordered by margin : \n" ,summary_to_print)
    print_matrix_in_file(summary_to_print_ordered_by_margin, linear_coeff_vs_kernel_ridge_margins_file, 'a')


    with open(linear_coeff_vs_kernel_ridge_margins_file,'a') as file:
        file.write('\n\n Ordered by linear regression coefficients, higher is better \n')
    X_header_and_numbers_T_order_by_linear_coef =  sorted(X_header_and_numbers_T, key=lambda kv: kv[3],  reverse=True) 
    print(" X header, margins and linear coef numbers after transpose, ordered by margin: \n", X_header_and_numbers_T_order_by_linear_coef)
    summary_header = np.array(["X_variable",  "meaning ", "kernel ridge margins", "linear regression coefficients", "difference"])
    summary_to_print_ordered_by_linear_coef = [summary_header, *X_header_and_numbers_T_order_by_linear_coef]
    print("margins and linearcoef summary_to_print ordered by linear regression coefficients : \n" ,summary_to_print_ordered_by_linear_coef)
    print_matrix_in_file(summary_to_print_ordered_by_linear_coef, linear_coeff_vs_kernel_ridge_margins_file, 'a')


    with open(linear_coeff_vs_kernel_ridge_margins_file,'a') as file:
        file.write('\n\n Ordered by absolute difference, between kernel ridge, and linear coefficients, the first has the maximum non linearity variation  \n')
    X_header_and_numbers_T_order_by_abs_diff =  sorted(X_header_and_numbers_T, key=lambda kv: kv[4],  reverse=True) 
    print(" X header, margins and linear coef numbers after transpose, ordered by margin: \n", X_header_and_numbers_T_order_by_abs_diff)
    summary_header = np.array(["X_variable",  "meaning ", "kernel ridge margins", "linear regression coefficients", "difference"])
    summary_to_print_ordered_by_abs_diff = [summary_header, *X_header_and_numbers_T_order_by_abs_diff]
    print("margins and linearcoef summary_to_print ordered by linear regression coefficients : \n" ,summary_to_print_ordered_by_abs_diff)
    print_matrix_in_file(summary_to_print_ordered_by_abs_diff, linear_coeff_vs_kernel_ridge_margins_file, 'a')





def capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice, 
                   d_X_i_linear_coefficients, 
                    file_path, X_meaning_dictionnary_ ):
    X_header = []
    X_variable_meanings = []
    for i in range(0, len(d_X_i_linear_coefficients)):
        X_header.append("X_"+ str(i))
        X_variable_meanings.append(X_meaning_dictionnary_["X_"+ str(i)])
    #coefficient_to_print = ["d_X_" + str(d_X_i_indice)] + d_X_i_linear_coefficients
    #coefficient_to_print =  np.insert(d_X_i_linear_coefficients.astype('|S10'), 0, "d_X_" + str(d_X_i_indice) )
    d_X_i_linear_coefficients_ = [float(x) for x in d_X_i_linear_coefficients]
    print(" --- FLOAT version of d_X_i_linear_coefficients_", repr(d_X_i_linear_coefficients_))
    X_header_and_numbers = []
    X_header_and_numbers.append(X_header)   
    X_header_and_numbers.append(X_variable_meanings)   
    X_header_and_numbers.append( [float(element) for element in d_X_i_linear_coefficients] )
    X_header_and_numbers.append([abs(float(element)) for element in d_X_i_linear_coefficients])
    X_header_and_numbers_T = np.array(X_header_and_numbers).T

   
    # we do this to force values to be float number as they should be for sorting to works well
    X_header_and_numbers_T = X_header_and_numbers_T.tolist()
    for j in range(2,4):
        for i in range(0,len(d_X_i_linear_coefficients)):
            print (" --- Before Modifying variable " +  X_header_and_numbers_T[i][j] + ", Type : " + str(type( X_header_and_numbers_T[i][j])))
            X_header_and_numbers_T[i][j] = float(X_header_and_numbers_T[i][j])
            print (" --- After Modifying variable " + str( X_header_and_numbers_T[i][j])+ ", Type : " + str(type( X_header_and_numbers_T[i][j])))


    summary_header = np.array(["Variable", "meaning ", "d_X_" + str(d_X_i_indice) + " (Variation relative to "+X_meaning_dictionnary_["X_"+ str(d_X_i_indice)] + ")",  "asolute d_X_" + str(d_X_i_indice) ] )
    summary_to_print = [summary_header, *X_header_and_numbers_T]
    print(" X header, d_X_ " + str(d_X_i_indice) + " values : \n", X_header_and_numbers)
    print_matrix_in_file(summary_to_print, file_path)

    with open(file_path,'a') as file:
        file.write('\n\n Ordered by value of coefficient, the first has the best positive interaction, with ' +X_meaning_dictionnary_["X_"+ str(d_X_i_indice)] +' \n ')
    X_header_and_numbers_T_order_by_coeff =  sorted(X_header_and_numbers_T, key=lambda kv: kv[2],  reverse=True) 
    print(" X header, margins and linear coef numbers after transpose, ordered by margin: \n", X_header_and_numbers_T_order_by_coeff)
    summary_header = np.array(["Variable", "meaning ", "d_X_" + str(d_X_i_indice) + " (Variation relative to "+X_meaning_dictionnary_["X_"+ str(d_X_i_indice)] + ")",  "asolute d_X_" + str(d_X_i_indice) ] )
    summary_to_print_ordered_by_coeff = [summary_header, *X_header_and_numbers_T_order_by_coeff]
    print("margins and linearcoef summary_to_print ordered by margin : \n" ,summary_to_print_ordered_by_coeff)
    print_matrix_in_file(summary_to_print_ordered_by_coeff, file_path, 'a')

    """
    print("--- Debugging bad ordering")
    print("--- Before Ordering : first element coefficient = " +  str(X_header_and_numbers_T[0][2]) + ", Type : " , type(X_header_and_numbers_T[0][2]))
    print("--- Before Ordering : second element coefficient = " + str( X_header_and_numbers_T[1][2]) + ", Type : ", type( X_header_and_numbers_T[1][2]))
    print("--- Before Ordering : third element coefficient = ",  str (X_header_and_numbers_T[2][2]) + ", Type : ", type( X_header_and_numbers_T[2][2]))
    print("--- Before Ordering : fourth element coefficient = " , str( X_header_and_numbers_T[3][2]) + ", Type: ", type( X_header_and_numbers_T[3][2]))

    print("--- Before Ordering : first superior to second:  as string, " 
                    + str ( X_header_and_numbers_T[0][2]> X_header_and_numbers_T[1][2] )
                    + ", As float " +  str( float(X_header_and_numbers_T[0][2]) > float(X_header_and_numbers_T[1][2]) ))
    print("--- Before Ordering : second superior to third:  as string, "     +   str(  X_header_and_numbers_T[1][2]> X_header_and_numbers_T[2][2]) 
                    + ", As float " +  str(  float(X_header_and_numbers_T[1][2]) > float(X_header_and_numbers_T[2][2])   ))
    print("--- Before Ordering : third superior to fourth:  as string, "     +  str(  X_header_and_numbers_T[2][2]> X_header_and_numbers_T[3][2] )
                    + ", As float " +   str( float(X_header_and_numbers_T[2][2]) > float(X_header_and_numbers_T[3][2])  ))

    
    print("--- after Ordering : first element coefficient = " + str(X_header_and_numbers_T_order_by_coeff[0][2]) + ", Type: ", type(X_header_and_numbers_T_order_by_coeff[0][2]))
    print("--- after Ordering : second element coefficient = " + str(X_header_and_numbers_T_order_by_coeff[1][2]) + ", Type: ", type(X_header_and_numbers_T_order_by_coeff[1][2]))
    print("--- after Ordering : third element coefficient = " + str(X_header_and_numbers_T_order_by_coeff[2][2]) + ", Type: ", type(X_header_and_numbers_T_order_by_coeff[2][2]))
    print("--- after Ordering : fourth element coefficient = " + str(X_header_and_numbers_T_order_by_coeff[3][2]) + ", Type: ", type(X_header_and_numbers_T_order_by_coeff[3][2]))


    print("--- After Ordering : first superior to second:  as string, "  +  str(  X_header_and_numbers_T_order_by_coeff[0][2]> X_header_and_numbers_T_order_by_coeff[1][2] )
                    + ", As float " +  str(  float(X_header_and_numbers_T_order_by_coeff[0][2]) > float(X_header_and_numbers_T_order_by_coeff[1][2])  ))
    print("--- After Ordering : second superior to third:  as string, "  +  str(  X_header_and_numbers_T_order_by_coeff[1][2]> X_header_and_numbers_T_order_by_coeff[2][2] )
                    + ", As float " +  str(  float(X_header_and_numbers_T_order_by_coeff[1][2]) > float(X_header_and_numbers_T_order_by_coeff[2][2])  ))
    print("--- After Ordering : third superior to fourth:  as string, "  +  str(  X_header_and_numbers_T_order_by_coeff[2][2]> X_header_and_numbers_T_order_by_coeff[3][2] )
                    + ", As float " +  str(  float(X_header_and_numbers_T_order_by_coeff[2][2]) > float(X_header_and_numbers_T_order_by_coeff[3][2])  ))
    """
    with open(file_path,'a') as file:
        file.write('\n\n Ordered by absolute value of coefficients,  the first has the best absolute interaction, with ' +X_meaning_dictionnary_["X_"+ str(d_X_i_indice)] +'  \n ')
    X_header_and_numbers_T_order_by_absolute_coeff =  sorted(X_header_and_numbers_T, key=lambda kv: kv[3],  reverse=True) 
    print(" X header, margins and linear coef numbers after transpose, ordered by margin: \n", X_header_and_numbers_T_order_by_absolute_coeff)
    summary_header = np.array(["Variable", "meaning ", "d_X_" + str(d_X_i_indice) + " (Variation relative to "+X_meaning_dictionnary_["X_"+ str(d_X_i_indice)] + ")",  "asolute d_X_" + str(d_X_i_indice) ] )
    summary_to_print_ordered_by_absolute_coef = [summary_header, *X_header_and_numbers_T_order_by_absolute_coeff]
    print("margins and linearcoef summary_to_print ordered by margin : \n" ,summary_to_print_ordered_by_absolute_coef)
    print_matrix_in_file(summary_to_print_ordered_by_absolute_coef, file_path, 'a')


 




def plot_marginal_effect_histogramm_graph(n_pointwise_margins, file, text_at_X = 0, text_at_Y = 0):
    import matplotlib.pyplot as plt
    import pandas
    plt.hist(n_pointwise_margins, 100)
    plt.axvline(n_pointwise_margins.mean(), color='k', linestyle='dashed', linewidth=2)
    plt.text(text_at_X, text_at_Y, str(pandas.DataFrame(n_pointwise_margins).describe()))
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=8)
    plt.savefig(file)
    plt.clf()
    plt.cla()
    plt.close()


import os
def get_data_from_summary_folder( summary_folder_path, data_to_extract = "configurations", format_ = "google_pixel_4a_5g_format", consider_exact_values_of_frequency =  False, X_format_in_model = "base_Y", maximum_input_size = -1):
    total_result_list = []
    list_of_files = {}
    print (" --- Getting data from folder  ", summary_folder_path)
    print (" --- Maximum input size =   ", maximum_input_size)
    print (" --- X format manipulated by the model  =   ", X_format_in_model)
    stop = 0
    if maximum_input_size != -1:
        for (dirpath, dirnames, filenames) in os.walk(summary_folder_path):
            for filename in filenames:
                #if filename.endswith('.html'): 
                #list_of_files[filename] = os.sep.join([dirpath, filename])
                current_result_list = get_data_from_summary_file(os.sep.join([dirpath, filename]), data_to_extract, format_, consider_exact_values_of_frequency, X_format_in_model)
                if len(total_result_list) + len(current_result_list) < maximum_input_size:
                    total_result_list = total_result_list + current_result_list
                else: 
                    total_result_list = total_result_list + current_result_list[0: maximum_input_size - len(total_result_list)]
                    stop = 1
                    break
            if stop == 1:
                break
    else:
        for (dirpath, dirnames, filenames) in os.walk(summary_folder_path):
            for filename in filenames:
                #if filename.endswith('.html'): 
                #list_of_files[filename] = os.sep.join([dirpath, filename])
                current_result_list = get_data_from_summary_file(os.sep.join([dirpath, filename]), data_to_extract, format_, consider_exact_values_of_frequency, X_format_in_model)
                total_result_list = total_result_list + current_result_list

    return total_result_list

# For now, we just implement functions for the google pixel. 
def get_data_from_summary_file( summary_file_path, data_to_extract = "configurations", format_ = "google_pixel_4a_5g_format", consider_exact_values_of_frequency = False, X_format_in_model = "base_Y"):
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
        line_data = line.split(",")

        # when the cc_info varied during experiments we give another chance to the configuration to be tested, we don't consider it
        #print("current line ", line_data)
        starting_cc_info = int(line_data[9])
        ending_cc_info =  int(line_data[10])
        if ending_cc_info - starting_cc_info != 0:
            continue

        if (data_to_extract == "configurations"):
            if (format_ == "human_readable_format"):
                current_configuration = line_data[0]
                result.append(current_configuration)
            else:
                if (format_ == "generic_format"):
                    if (not consider_exact_values_of_frequency):
                        current_configuration = line_data[1]
                    else:
                        current_configuration = line_data[2]
                elif (format_ == "google_pixel_4a_5g_format"): # or  can be"samsung_galaxy_s8_format", or "generic"
                    if (not consider_exact_values_of_frequency):
                        if(X_format_in_model == "base_Y"):
                            # In the case where we cannot longer set different values of frequencies to core on the same socket; 
                            # variables are no longer independant, 
                            # So we introduce another notation factorizing the not nul frequency level to the first newly introduced position
                            current_configuration = line_data[3]
                            current_configuration = (current_configuration[1:-1]).replace(" ", "").split("-")
                            #print ("arrays of value as string : ", current_configuration)
                            X_line = [float(numeric_string) for numeric_string in current_configuration]
                            X_line = convert_from_configuration_to_base_Y(X_line)
                            #print("--- Special case : resulted X configuration in base Y format: ", X_line)
                            result.append(X_line)
                            continue
                        elif (X_format_in_model == "base_Y_N_on_socket"):
                            current_configuration = line_data[3]
                            current_configuration = (current_configuration[1:-1]).replace(" ", "").split("-")
                            #print ("arrays of value as string : ", current_configuration)
                            X_line = [float(numeric_string) for numeric_string in current_configuration]
                            X_line = convert_from_configuration_to_base_Y_N_on_socket(X_line)
                            #print("--- Special case : resulted X configuration in base Y format: ", X_line)
                            result.append(X_line)
                            continue
                        elif(X_format_in_model == "base_Y_F"):
                            # More than Base_Y,  foreach socket another variable has been introduced concerning the fact that the frequency of the socket is modify or no
                            current_configuration = line_data[3]
                            current_configuration = (current_configuration[1:-1]).replace(" ", "").split("-")
                            #print ("arrays of value as string : ", current_configuration)
                            X_line = [float(numeric_string) for numeric_string in current_configuration]
                            X_line = convert_from_configuration_to_base_Y_F(X_line)
                            #print("--- Special case : resulted X configuration in base Y format: ", X_line)   ### base_Y_F_N_on_socekt
                            result.append(X_line)
                            continue
                        elif(X_format_in_model == "base_Y_F_N_on_socket"):
                            # More than Base_Y,  foreach socket another variable has been introduced concerning the fact that the frequency of the socket is modify or no
                            current_configuration = line_data[3]
                            current_configuration = (current_configuration[1:-1]).replace(" ", "").split("-")
                            #print ("arrays of value as string : ", current_configuration)
                            X_line = [float(numeric_string) for numeric_string in current_configuration]
                            X_line = convert_from_configuration_to_base_Y_F_N_on_socket(X_line)
                            #print("--- Special case : resulted X configuration in base Y format: ", X_line)   ### base_Y_F_N_on_socket
                            result.append(X_line)
                            continue
                        else:
                            current_configuration = line_data[3]
                elif (format_ == "samsung_galaxy_s8_format"):
                    if (not consider_exact_values_of_frequency):
                        if(X_format_in_model == "base_Y"):
                            # In the case where we cannot longer set different values of frequencies to core on the same socket; 
                            # variables are no longer independant, 
                            # So we introduce another notation factorizing the not nul frequency level to the first newly introduced position
                            current_configuration = line_data[3]
                            current_configuration = (current_configuration[1:-1]).replace(" ", "").split("-")
                            #print ("arrays of value as string : ", current_configuration)
                            X_line = [float(numeric_string) for numeric_string in current_configuration]
                            X_line = convert_from_configuration_to_base_Y(X_line, format_)
                            #print("--- Special case : resulted X configuration in base Y format: ", X_line)
                            result.append(X_line)
                            continue
                        elif (X_format_in_model == "base_Y_N_on_socket"):
                            current_configuration = line_data[3]
                            current_configuration = (current_configuration[1:-1]).replace(" ", "").split("-")
                            #print ("arrays of value as string : ", current_configuration)
                            X_line = [float(numeric_string) for numeric_string in current_configuration]
                            X_line = convert_from_configuration_to_base_Y_N_on_socket(X_line, format_)
                            #print("--- Special case : resulted X configuration in base Y format: ", X_line)
                            result.append(X_line)
                            continue
                        else:
                            current_configuration = line_data[3]
                    else:
                        current_configuration = line_data[4]
                
                current_configuration = (current_configuration[1:-1]).replace(" ", "").split("-")
                #print ("arrays of value as string : ", current_configuration)
                X_line = [float(numeric_string) for numeric_string in current_configuration]
                #print("resulted X configuration : ", X_line)
                result.append(X_line)
        elif (data_to_extract == "energy_efficiency"):
            current_value = line_data[8]
            y_line_numeric = float(current_value)
            y_line_numeric_energy_efficiency =  1 / y_line_numeric
            #print("resulted y value : ", y_line_numeric)
            result.append(y_line_numeric_energy_efficiency)
            #print("after adding it to y : ", y_line_numeric)
        elif (data_to_extract == "energy"):
            current_value = line_data[5]
            y_line_numeric = float(current_value)
            #print("resulted y value : ", y_line_numeric)
            result.append(y_line_numeric)
            #print("after adding it to y : ", y_line_numeric)
        elif (data_to_extract == "power"):
            current_value = line_data[6]
            y_line_numeric = float(current_value)
            #print("resulted y value : ", y_line_numeric)
            result.append(y_line_numeric)
            #print("after adding it to y : ", y_line_numeric)
        elif (data_to_extract == "workload"):
            current_value = line_data[7]
            y_line_numeric = float(current_value)
            #print("resulted y value : ", y_line_numeric)
            result.append(y_line_numeric)
            #print("after adding it to y : ", y_line_numeric)
    return result           
            




def remove_aberrant_points(X_user_friendly, X, y, energy_array, workload_array, energy_gap, number_of_neighbour, sigma_2, repeat_experiments):
    # This function cleans datas by 
    # Removing some points where we suspect that the energy measurement was not good

    X_removed_indexes = []
    final_X_user_friendly = []
    final_X = []
    final_y = []
    final_energy_array = []
    final_workload_array = []

    if repeat_experiments:
        global distance_matrix_raw_by_raw_on_X
        # This matrix, for each couple of observations (i, j) saves the value of || X [i, :] - X[j, :] ||
        distance_matrix_raw_by_raw_on_X= []
        global exp_matrix
        # this matrix, for each couple of observations (i,j) , saves the value of  exp (-|| X [i, :] - X[j, :] ||/ sigma_2)
        # it is the k(i,j) matrix. 
        exp_matrix = []

    index_of_loop = 0
    for val in X:
        print(" --- remove_aberrant_points: do we remove value ", val)
      
        is_aberrant = is_aberrant_point(index_of_loop, X_user_friendly, np.asarray(X), y, energy_array, workload_array, energy_gap, number_of_neighbour, sigma_2)
        if (not is_aberrant):
            print(" --- remove_aberrant_points: The value " + repr(val) + " is not an abberant point.")
            final_X_user_friendly.append(X_user_friendly[index_of_loop])
            final_X.append(val)
            final_y.append(y[index_of_loop])
            final_energy_array.append(energy_array[index_of_loop])
            final_workload_array.append(workload_array[index_of_loop])
        else:
            print("--- remove_aberrant_points: The value " + repr(val) + " is  an abberant point. we don't add it")
           
            X_removed_indexes.append(index_of_loop)
        
        index_of_loop = index_of_loop +1
    print("--- remove_aberrant_points: Printing all "+ repr(len(X_removed_indexes))+" removed points ")
    for x_removed_index in X_removed_indexes:
        print(" --- Configuration: " , X_user_friendly[x_removed_index])
        print(" --- Energy: " , energy_array[x_removed_index])
    print ("final_X_user friendly : \n ", final_X_user_friendly)
    print ("final_X : \n ", final_X)
    print ("final_y : \n ", final_y)
    return final_X_user_friendly, final_X, final_y

    





# I reuse some function used to compute the marginal effect 
from scipy import spatial as ssp
# This matrix, for each couple of observations (i, j) saves the value of || X [i, :] - X[j, :] ||
distance_matrix_raw_by_raw_on_X= []

# this matrix, for each couple of observations (i,j) , saves the value of  exp (-|| X [i, :] - X[j, :] ||/ sigma_2)
# it is the k(i,j) matrix. 
exp_matrix = []


# For an observation i function to compute  exp_terms (i), exp_terms (i) = vector of n values, where each is  [exponential(- square_norm(X[n,:] , X[i:])/ sigma_2)]
def exponential_minus_sigma2(value_in_array, sigma_2):
    result = math.exp(-1*value_in_array/sigma_2)
    return result 


def only_exp_terms(X, i, sigma_2):
    result = []
    N = len(X)
    #print ("** START computing exp_terms on observation ", i)
    #print ("** X = ", X)
    global distance_matrix_raw_by_raw_on_X
    if len(distance_matrix_raw_by_raw_on_X) == 0 :
        distance_matrix_raw_by_raw_on_X = ssp.distance.cdist(X[:,:], X[:,:], 'minkowski', p=2.)
    #print("distance_matrix_raw_by_raw_on_X : ", distance_matrix_raw_by_raw_on_X)
    vector_of_squares_of_norms_of_diff_vector_of_X_n_lines_centered_to_ith_line = []
    for n in range(0,N):
        #print("X["+str(i)+",:] = ")
        #print(X[i,:])
        #print("X["+str(n)+",:] = ", X[n,:]) 
        #current_square_of_norm =  ( ssp.distance.euclidean(X[n,:], X[i,:]) )**2
        #print("distance_matrix_raw_by_raw_on_X [" + str(n) + "," + str(i) +" ]: ", ssp.distance.euclidean(X[n,:], X[i,:]) )
        current_square_of_norm = distance_matrix_raw_by_raw_on_X[n,i] ** 2
        vector_of_squares_of_norms_of_diff_vector_of_X_n_lines_centered_to_ith_line.append(current_square_of_norm)
    vexponential_minus_x_by_sigma2 = np.vectorize(exponential_minus_sigma2)                     
    result = vexponential_minus_x_by_sigma2(vector_of_squares_of_norms_of_diff_vector_of_X_n_lines_centered_to_ith_line, [sigma_2]*N )[:,np.newaxis]
    #print ("** END computing exp_terms on observation " + str(i) + ", result = " + str(result))
    return np.array(result);                 


# Computing the vector exp_matrix where exp_matrix has N lines and at line n we have the transposed of the vector exp_term(i)
# in the paper image where I did the demonstration, it is called Diag
def get_exp_matrix(X, sigma_2):
    print ("*** START computing ci exp matrix ")
    print ("X = ", X)
    global exp_matrix
    if len(exp_matrix)== 0:
        exp_matrix = []
        N = len(X)
        for i in range (0, N):
            exp_matrix.append(only_exp_terms(X, i, sigma_2))
        print ("*** END computing ci exp matrix, first computation result ", np.array(exp_matrix))
    else:
        print ("*** END computing ci exp matrix, cached  result ", np.array(exp_matrix))
    return np.array(exp_matrix)




def is_aberrant_point(point_position, X_user_friendly, X, y, energy_array, workload_array, energy_gap, number_of_neighbour, sigma_2):
    # This is the algorithmic is to test is a data point is abberant or not
    # The first step is to obtain a list of the first neighbour of the datadapoint, 
    #       Neighnour are computed regarding the gaussian kernel distance between data-point 
    #       the number of neighbours (number_of_neighbour) is a parameter of this function
    # The second step is to compute the "median" of the energy consumption of all this neighour
    # The third step is to compute the difference beetween this median computed above 
    #       and the energy consomption of the current data-point. 
    #       and if this distance is above a certain threshold the data-point is simply removed from the dataset. 
    #       The threshold (energy_gap) is also a paramter to this function

    print("--- Computing the list of the " + repr(number_of_neighbour) + " first neighbours of " + repr(X_user_friendly[point_position]))
    kernel_matrix =  get_exp_matrix(X, sigma_2)
    print("--- X size ", len(X))
    print("--- kernel matrix size ", len(kernel_matrix))
    kernel_distance_array = kernel_matrix [point_position]
    sorted_kernel_distance_dictionnary = sorted(enumerate(kernel_distance_array), key=lambda kv: kv[1],  reverse=True) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
    neighbours_dictionnary = []                                                                                        # Notice that the kernel value is between 1 and 0, 1 means close and 0 means too far
    print("--- Ordered by distance, Printing the list of the " + repr(number_of_neighbour) + " first neighbours of " + repr(X_user_friendly[point_position]))
    for index in range(0,number_of_neighbour): 
        if (index < len(sorted_kernel_distance_dictionnary) ):                                            
            position_in_data_point = sorted_kernel_distance_dictionnary[index][0]  
            print("--- Neighbour  " + str(index) + " in the list of neghbours, And at position " + repr(position_in_data_point) + " in the X datas point")   
            print("--------------")
            print(" --- Configuration: " , X_user_friendly[position_in_data_point] )
            print(" --- Distance from that configuration: ", sorted_kernel_distance_dictionnary[index][1]  )
            print(" --- Energy efficiency: ", y[position_in_data_point] )
            print(" --- Energy: " , energy_array[position_in_data_point] )
            print(" --- Workload: ", workload_array[position_in_data_point] )
            print("--------------")
            current_triplet_position_distance_energy = [sorted_kernel_distance_dictionnary[index][0], sorted_kernel_distance_dictionnary[index][1]]
            current_triplet_position_distance_energy.append(energy_array[position_in_data_point] )  # appending to energy to get the median of the energy, not of the kernel distance 
            neighbours_dictionnary.append(current_triplet_position_distance_energy)
    
    print("--- Ordered by energy, Printing the list of the " + repr(number_of_neighbour) + " first neighbours of " + repr(X_user_friendly[point_position]))
    neighbours_dictionnary_sorted_by_energy = sorted(neighbours_dictionnary, key=lambda kv: kv[2],  reverse=False)
    for index in range(0,number_of_neighbour): 
        if (index < len(neighbours_dictionnary_sorted_by_energy) ):                                            
            position_in_data_point = neighbours_dictionnary_sorted_by_energy[index][0]  
            print("--- Neighbour  " + str(index) + " in the list of neghbours, And at position " + repr(position_in_data_point) + " in the X datas point")   
            print("--------------")
            print(" --- Configuration: " , X_user_friendly[position_in_data_point] )
            print(" --- Distance from that configuration: ", sorted_kernel_distance_dictionnary[index][1]  )
            print(" --- Energy efficiency: ", y[position_in_data_point] )
            print(" --- Energy: " , energy_array[position_in_data_point] )
            print(" --- Workload: ", workload_array[position_in_data_point] )
            print("--------------")
           

    median_index_in_the_neighbours_dictionnary = int(number_of_neighbour/2) - 1
    median_triplet_in_the_neighbours_dictionnary = neighbours_dictionnary_sorted_by_energy[median_index_in_the_neighbours_dictionnary]       #  Can obtain something like (4, dist_3)
    median_position_in_data_point = median_triplet_in_the_neighbours_dictionnary[0]
    print("--------------")
    print("--- Median at position " + repr(median_index_in_the_neighbours_dictionnary) + " in the list of neghbours, And at position " + repr(median_position_in_data_point) + " in the X datas point")   
    print("--------------")
    print(" --- Configuration: " , X_user_friendly[median_position_in_data_point] )
    print(" --- Energy efficiency: ", y[median_position_in_data_point] )
    print(" --- Energy: " , energy_array[median_position_in_data_point] )
    print(" --- Workload: ", workload_array[median_position_in_data_point] )
    print("--------------")

    print("--- Comparing the median energy with the energy of that data point")
    median_energy = energy_array[median_position_in_data_point] 
    if (abs(median_energy - energy_array[point_position]) > energy_gap): 
        print("--- The energy of the current configuration (" + repr(energy_array[point_position]) + " mAh)  is far from the median.")
        print("---  Median :" + repr(median_energy) + ",   the gap is :  " + repr(energy_gap) )
        print("--- So yes we remove this configuration " + repr(X_user_friendly[point_position]) )
        return True
    print("--- The energy of the current configuration (" + repr(median_energy) + " mAh)  it is NOT far from the median.")
    print("---  Median :" + repr(median_energy) + ",   the gap is :  " + repr(energy_gap) )
    print("--- So No we don't romove this configuration " + repr(X_user_friendly[point_position]) )
    return False

def count_number_of_input_with_fourth_core_on(X_train):
    counter = 0
    for x in X_train:
        if x[4] == 1:
            counter = counter + 1
    return counter

def inputs_where_d_X_index_is_negative(pointwise_margins, variable_index, X_train):
    result = []
    index = 0
    for x in X_train:
        if (pointwise_margins[index, variable_index] < 0):
            result.append(x)
        index = index + 1
    return result

def element_in_list(element, list_):
    indices = []
    for i in range(len(list_)):
        if list_[i] == element:
            indices.append(i)
        
    return indices

def aggregate_datas_old(X_array, y_array):
    # this function takes as input two arrays
    # it for all pair x, y , one value on x and another on y
    # if a particular value of x appears many times
    # it return pairs x, m  when m is the mean of all y values associated to x
    # It is a sort of aggragation based on x, with the mean
    # the resulted array are in the form of X_array (with disjoints points) and m_array (means associated to each y) 
    index_of_loop = 0
    X_result = []
    m_result = []
    registered_duplicates = []
    sorted_x_y_result = []
    for x in X_array:
        #print(" --- Checking value ", x)
        #print(" --- Retained values ", X_result)
        final_places = element_in_list(x, X_result)
        if  len(final_places) == 0:
            X_result.append(x)
            m_result.append(y_array[index_of_loop])
            #print(" --- Answer : we add the value, it is  not yet present")  
        else :# (is_from_manual_experiment(X_user_friendly[index_of_loop])):
            #print(" --- Answer : element is present, have it been processed, if yes , indices in registered duplicates ", element_in_list(x, registered_duplicates))  
            if (len(element_in_list(x, registered_duplicates)) == 0) : # no
                initial_places = element_in_list(x, X_array)
                #print(" --- Answer : the element " + repr(X_array[index_of_loop]) +" is present at positions " + repr(initial_places) )
                position_y_array = []
                for duplicate_index in initial_places:
                    #print(" --- Position: " , duplicate_index )
                    #print("--------------")
                    #print(" --- x: " , X_array[duplicate_index] )
                    #print(" --- y: " , y_array[duplicate_index] )
                    position_y_array.append([duplicate_index, y_array[duplicate_index]])
                    #print("--------------")

                # Now getting the mean of duplicates, regarding the y value
                sorted_position_y_array = sorted(position_y_array, key=lambda kv: kv[1],  reverse=True) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
                #print("---------------------- Listing and computing the mean")
                #print("--- Ordered by y, #printing the list of the " + str(len(initial_places)) + " duplicates of " + repr(X_array[index_of_loop]))
                mean_y = 0
                for index in range(0,len(initial_places)):                                     
                    position_in_data_point = sorted_position_y_array[index][0]  
                    #print("---  Duplicate  " + str(index) + " in the list of duplicate, And at position " + repr(position_in_data_point) + " in the X datas point")   
                    #print("--------------")
                    #print(" --- x: " , X_array[position_in_data_point] )
                    #print(" --- y: ", y_array[position_in_data_point] )
                    #print("--------------")
                    mean_y = mean_y +  y_array[position_in_data_point]
                   
                mean_y = mean_y / len(initial_places) 
                #print("--------------")
                #print("--- We will append this mean as duplicate reprensentant in the X datas point")   
                #print("--------------")
                #print(" --- x: ", X_array[position_in_data_point] )
                #print(" --- y: ", mean_y )
                #print("--------------")
                """
                X_result[final_places[0]] = X_array[position_in_data_point]
                m_result[final_places[0]] = mean_y
                """
                sorted_x_y_result.append([X_array[position_in_data_point], mean_y])
                registered_duplicates.append(x)
        
        index_of_loop = index_of_loop +1
    sorted_x_y_result = sorted(sorted_x_y_result, key=lambda kv: kv[0],  reverse=True) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
    # emptying X_result and m_result
    X_result = []
    m_result = []
    for couple_x_y in sorted_x_y_result:
        #print("--- We append this mean as duplicate reprensentant in the X datas point")   
        #print("--------------")
        #print(" --- x: ", couple_x_y[0] )
        #print(" --- y: ", couple_x_y[1] )
        #print("--------------")
        X_result.append(couple_x_y[0])
        m_result.append(couple_x_y[1])

    return X_result, m_result





def aggregate_datas(X_array, y_array):
    # this function takes as input two arrays
    # it for all pair x, y , one value on x and another on y
    # if a particular value of x appears many times
    # it return pairs x, m  when m is the mean of all y values associated to x
    # It is a sort of aggragation based on x, with the mean
    # the resulted array are in the form of X_array (with disjoints points) and m_array (means associated to each y) 
    X_result = []
    m_result = []
    grouped_x = []
    sorted_x_y_result = []
    for x, y in zip(X_array, y_array):
        print(" --- Checking value ", x)
        print(" --- Retained values ", X_result)
        x_element_classified = element_in_list(x, X_result)
        if  len(x_element_classified) == 0:
            X_result.append(x)
            m_result.append(y)
            print(" --- Answer : we add the value, it was not yet present")  
        else : 
            print(" --- Answer : element is present, have it been processed ", element_in_list(x, grouped_x))  
            if (len(element_in_list(x, grouped_x)) == 0) : # no
                x_positions_in_input_data = element_in_list(x, X_array)
                #print(" --- Answer : the element " + x +" is present at positions " + repr(x_positions_in_input_data) )
                
                mean_y = 0
                for x_position in x_positions_in_input_data:                                     
                    mean_y = mean_y +  y_array[x_position]
                   
                mean_y = mean_y / len(x_positions_in_input_data) 
                #print("--------------")
                #print("--- We will append this mean as duplicate reprensentant in the X datas point")   
                #print("--------------")
                #print(" --- x: ", x)
                #print(" --- y: ", mean_y )
                #print("--------------")
                sorted_x_y_result.append([X_array[x_position], mean_y])
                grouped_x.append(x)
        
       
    sorted_x_y_result = sorted(sorted_x_y_result, key=lambda kv: kv[0],  reverse=False) # with original indexes like [ (12, dist_1), (0, dist_2), (4, dist_3)..  ]
    # emptying X_result and m_result
    X_result = []
    m_result = []
    for couple_x_y in sorted_x_y_result:
        #print("--- We append this mean as duplicate reprensentant in the X datas point")   
        #print("--------------")
        #print(" --- x: ", couple_x_y[0] )
        #print(" --- y: ", couple_x_y[1] )
        #print("--------------")
        X_result.append(couple_x_y[0])
        m_result.append(couple_x_y[1])

    return X_result, m_result

def decapitalize(str):
    return str[:1].lower() + str[1:]

def remove_abbreviation(str):
    return str.replace("freq.", "frequency (freq.)" )


def plot_one_marginal_plot(fig, X_train, pointwise_margins, indice_name, cibled_indice, X_meaning_dictionnary_, transparency,  workstep = "", paper_fontsize = 12, acceptable_marginal_mean_value = 0):
    round_points = plt.scatter(X_train[:,indice_name], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "black")
    for current_line_index in range(0, 50):
        #print("--- Nothing, just to avoid indentation error")
        plt.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_name]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "black")
    x_aggregated, y_means = aggregate_datas(X_train[:,indice_name], pointwise_margins[:,cibled_indice])
    plt.plot(x_aggregated, y_means ,  color='black', marker='o', linestyle='-' ) #,markersize=2000 , linewidth = 10) #'-', '--', '-.', ':', ''
    #plt.axhline(y = 0, color = '#993300', linestyle = 'dashed') 
    #plt.axhline(y = 1, color = 'black', linestyle = 'dashed') 
    plt.axhline(y = acceptable_marginal_mean_value, color = "red" , linestyle = 'dashed') #'#ff9900'
    
    if acceptable_marginal_mean_value != 0:
        print("---- Acceptable marginal mean value = ", acceptable_marginal_mean_value)
        plt.axhline(y = acceptable_marginal_mean_value, color = "#ff9900" , linestyle = 'dashed', ) #
    
    
    if  workstep == "plotting_graphs_for_the_paper" or workstep == "computing_static_dynamic_score_for_paper" or  workstep == "finding_best_input_dataset_size":
        plt.ylabel( "d_energy_efficiency" + \
                                 "/\nd_" + X_meaning_dictionnary_["X_"+ str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  + ")", fontsize = paper_fontsize)
        plt.yticks(fontsize=paper_fontsize)
        plt.xlabel( X_meaning_dictionnary_["X_"+ str(indice_name)] + \
                                        " (" +"X_" + str(indice_name) + ")",  fontsize = paper_fontsize )
        
        plt.xlim(xmin=-0.25)
        plt.xticks(fontsize=paper_fontsize)
        ax = plt.gca()
        ax.yaxis.get_offset_text().set_fontsize(paper_fontsize)
        plt.locator_params(axis='x', nbins=4)

        
    if  workstep == "plotting_graphs_for_the_paper" or workstep == "computing_static_dynamic_score_for_paper" or  workstep == "finding_best_input_dataset_size":
        result=[]
        result.append(x_aggregated)
        result.append(y_means)
        return result



def plot_single_marginal_plot(name_in_global_plot, X_train, pointwise_margins, indice_name, cibled_indice, X_meaning_dictionnary_, transparency,  workstep = "", paper_fontsize = 12, acceptable_marginal_mean_value = 0):
    name_in_global_plot.scatter(X_train[:,indice_name], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "black")
    for current_line_index in range(0, 50):
        #print("--- Nothing, just to avoid indentation error")
        name_in_global_plot.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_name]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "black")
    x_aggregated, y_means = aggregate_datas(X_train[:,indice_name], pointwise_margins[:,cibled_indice])
    name_in_global_plot.plot(x_aggregated, y_means ,  color='black', marker='o', linestyle='-' ) #,markersize=2000 , linewidth = 10) #'-', '--', '-.', ':', ''
    #name_in_global_plot.axhline(y = 0, color = '#993300', linestyle = 'dashed') 
    #name_in_global_plot.axhline(y = 1, color = 'black', linestyle = 'dashed') 
    name_in_global_plot.axhline(y = acceptable_marginal_mean_value, color = "#800000" , linestyle = 'dashed') #'#ff9900'
    
    if acceptable_marginal_mean_value != 0:
        print("---- Acceptable marginal mean value = ", acceptable_marginal_mean_value)
        name_in_global_plot.axhline(y = acceptable_marginal_mean_value, color = "#ff9900" , linestyle = 'dashed', ) #
    
    
    if  workstep == "plotting_graphs_for_the_paper" or workstep == "computing_static_dynamic_score_for_paper" or  workstep == "finding_best_input_dataset_size":
        #name_in_global_plot.set_ylabel( r'$\partial f$'  + "/" + \
        #                         r'$\partial $' + "(" + X_meaning_dictionnary_["X_"+ str(indice_name)] + ")", fontsize = paper_fontsize)
        name_in_global_plot.set_ylabel(  "d_f/" + \
                                   "d_" + X_meaning_dictionnary_["X_"+ str(cibled_indice)] + " (" +"X_" + str(cibled_indice) + ")", fontsize = paper_fontsize)
        name_in_global_plot.tick_params(axis='x', which='major' , labelsize= paper_fontsize)
        name_in_global_plot.tick_params(axis='y', which='major' , labelsize= paper_fontsize/2)
        name_in_global_plot.tick_params(size=8)
        name_in_global_plot.set_xlabel( X_meaning_dictionnary_["X_"+ str(indice_name)] + \
                                        " (" +"X_" + str(indice_name) + ")" ,  fontsize = paper_fontsize )
        name_in_global_plot.set_xlim(xmin=-0.25)
        name_in_global_plot.yaxis.get_offset_text().set_fontsize(paper_fontsize)
    else:
        name_in_global_plot.set_title(  "Pointwise marginal effect with respect to the \n  " + decapitalize(X_meaning_dictionnary_["X_" + str(cibled_indice)]) + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                        " over \n " + X_meaning_dictionnary_["X_" + str(indice_name)] +   " (" +  "X_" + str(indice_name)  +")")
        name_in_global_plot.tick_params(size=8)
        #name_in_global_plot.set_xlabel( X_meaning_dictionnary_["X_"+ str(indice_name)] + "(" +"X_" + str(indice_name) + ")" )
        name_in_global_plot.set_xlim(xmin=-0.25)
        
    if workstep == "plotting_graphs_for_the_paper" or workstep == "computing_static_dynamic_score_for_paper" or  workstep == "finding_best_input_dataset_size":
        result=[]
        result.append(x_aggregated)
        result.append(y_means)
        return result
    



def plot_marginal_interactions (X_train, pointwise_margins, cibled_indice, indice_0, indice_1, indice_2,  indice_3, 
           indice_4, indice_5, indice_6, indice_7, indice_8 , X_meaning_dictionnary_, marginal_effect_exploration_folder_, workstep =  "" , paper_fontsize = 12, repeat_experiments = False):

    transparency = 0.007
    J_L_mapping = []
    if  workstep == "plotting_graphs_for_the_paper" or workstep == "computing_static_dynamic_score_for_paper" or  workstep == "finding_best_input_dataset_size":

        one_fig = plt.figure()
        
        print("--- Just for test " + decapitalize( X_meaning_dictionnary_["X_" + str(cibled_indice)] ) + " (" +  "X_" + str(cibled_indice)  +")" +  " and " + \
         decapitalize( X_meaning_dictionnary_["X_" + str(indice_0)] ) + " (" +  "X_" + str(indice_0)  + ").")

        one_plot_mapping = plot_one_marginal_plot( one_fig, X_train, pointwise_margins, indice_0, cibled_indice, X_meaning_dictionnary_, transparency, workstep, 16)

        # Add title and axis names
        plt.title( "Interaction between " +  decapitalize( X_meaning_dictionnary_["X_" + str(cibled_indice)] ) +  " and \n " + \
                                                    remove_abbreviation(decapitalize( X_meaning_dictionnary_["X_" + str(indice_0)] )) + "", fontsize = 16)
                                                     # (" +  "X_" + str(indice_0)  + ")
       
        plt.tight_layout()       

        if not repeat_experiments:
            plt.savefig(marginal_effect_exploration_folder_ + "/"+  X_meaning_dictionnary_["X_" + str(cibled_indice)].replace(" ","_")[-16:] + \
                                              "_over_" +  X_meaning_dictionnary_["X_" + str(indice_0)].replace(" ","_")[-16:] + ".png")

        plt.savefig(marginal_effect_exploration_folder_ + "/"+  "X_" + str(cibled_indice) + \
                                              "_over_" + "X_" + str(indice_0) + "__" +  ".png")

        plt.clf()
        plt.cla()
        plt.close()






        print("--- In function plot_marginal_interactions : plotting d_X_" + str(cibled_indice) + " with regard to X_"  + str(indice_0) + ", X_" + str(indice_1) + ", X_" + str(indice_2)+ ", ... X_" + str(indice_8))
        fig, ((d_X_cibled_indice_over_X_indice_1, d_X_cibled_indice_over_X_indice_2, d_X_cibled_indice_over_X_indice_3, d_X_cibled_indice_over_X_indice_4  ) ,                                                                                                                                                             #sharex = False, sharey=True,   width, height
          ( d_X_cibled_indice_over_X_indice_5, d_X_cibled_indice_over_X_indice_6, d_X_cibled_indice_over_X_indice_7, d_X_cibled_indice_over_X_indice_8 )) = plt.subplots(nrows= 2, ncols = 4,  figsize=(45, 17))

    
        J_L_mapping.append(one_plot_mapping)
        J_L_mapping.append(plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_1, X_train, pointwise_margins, indice_1, cibled_indice, X_meaning_dictionnary_, transparency, workstep, paper_fontsize))
        J_L_mapping.append(plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_2, X_train, pointwise_margins, indice_2, cibled_indice, X_meaning_dictionnary_, transparency, workstep, paper_fontsize))
        J_L_mapping.append(plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_3, X_train, pointwise_margins, indice_3, cibled_indice, X_meaning_dictionnary_, transparency, workstep, paper_fontsize))
        J_L_mapping.append(plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_4, X_train, pointwise_margins, indice_4, cibled_indice, X_meaning_dictionnary_, transparency, workstep, paper_fontsize))
        J_L_mapping.append(plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_5, X_train, pointwise_margins, indice_5, cibled_indice, X_meaning_dictionnary_, transparency, workstep, paper_fontsize))
        J_L_mapping.append(plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_6, X_train, pointwise_margins, indice_6, cibled_indice, X_meaning_dictionnary_, transparency, workstep, paper_fontsize))
        J_L_mapping.append(plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_7, X_train, pointwise_margins, indice_7, cibled_indice, X_meaning_dictionnary_, transparency, workstep, paper_fontsize))
        J_L_mapping.append(plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_8, X_train, pointwise_margins, indice_8, cibled_indice, X_meaning_dictionnary_, transparency, workstep, paper_fontsize))


      
    else :
        print("--- In function plot_marginal_interactions : plotting d_X_" + str(cibled_indice) + " with regard to X_"  + str(indice_0) + ", X_" + str(indice_1) + ", X_" + str(indice_2)+ ", ... X_" + str(indice_8))
        fig, ((d_X_cibled_indice_over_X_indice_0, d_X_cibled_indice_over_X_indice_3, d_X_cibled_indice_over_X_indice_6  ) , 
          (d_X_cibled_indice_over_X_indice_1, d_X_cibled_indice_over_X_indice_4, d_X_cibled_indice_over_X_indice_7 ),              #sharex = False, sharey=True,   width, height
          (d_X_cibled_indice_over_X_indice_2, d_X_cibled_indice_over_X_indice_5, d_X_cibled_indice_over_X_indice_8 )) = plt.subplots(nrows= 3, ncols = 3,  figsize=(22, 17))

    
      
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_0, X_train, pointwise_margins, indice_0, cibled_indice, X_meaning_dictionnary_, transparency, workstep)
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_1, X_train, pointwise_margins, indice_1, cibled_indice, X_meaning_dictionnary_, transparency, workstep)
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_2, X_train, pointwise_margins, indice_2, cibled_indice, X_meaning_dictionnary_, transparency, workstep)
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_3, X_train, pointwise_margins, indice_3, cibled_indice, X_meaning_dictionnary_, transparency, workstep)
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_4, X_train, pointwise_margins, indice_4, cibled_indice, X_meaning_dictionnary_, transparency, workstep)
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_5, X_train, pointwise_margins, indice_5, cibled_indice, X_meaning_dictionnary_, transparency, workstep)
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_6, X_train, pointwise_margins, indice_6, cibled_indice, X_meaning_dictionnary_, transparency, workstep)
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_7, X_train, pointwise_margins, indice_7, cibled_indice, X_meaning_dictionnary_, transparency, workstep)
        plot_single_marginal_plot( d_X_cibled_indice_over_X_indice_8, X_train, pointwise_margins, indice_8, cibled_indice, X_meaning_dictionnary_, transparency, workstep)

    #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")
    picture_name =  X_meaning_dictionnary_["X_" + str(cibled_indice)].replace(" ","_")[-16:] + \
                                              "_over_" +  X_meaning_dictionnary_["X_" + str(indice_0)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_2)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_3)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_4)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_5)].replace(" ","_")[-16:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_6)].replace(" ","_")[-16:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_7)].replace(" ","_")[-16:] + "__" + \
                                         X_meaning_dictionnary_["X_" + str(indice_8)].replace(" ","_")[-16:]  + ".png" 

    reduced_picture_name =  "X_" + str(cibled_indice) + \
                                              "_over_" + "X_" + str(indice_0) + "__" + \
                                           "X_" + str(indice_1)  + "__" + \
                                            "X_" + str(indice_2) + "__" + \
                                            "X_" + str(indice_3) + "__" + \
                                            "X_" + str(indice_4) + "__" + \
                                            "X_" + str(indice_5) + "__" + \
                                             "X_" + str(indice_6) + "__" + \
                                              "X_" + str(indice_7) + "__" + \
                                             "X_" + str(indice_8)  + ".png" 

    picture_title =  X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"\
                                              " over " +  X_meaning_dictionnary_["X_" + str(indice_0)] +   " (" +  "X_" + str(indice_0)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)] + " (" +  "X_" + str(indice_1)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_2)] + " (" +  "X_" + str(indice_2)  +"), \n"  + \
                                            X_meaning_dictionnary_["X_" + str(indice_3)] +  " (" +  "X_" + str(indice_3)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_4)] +  " (" +  "X_" + str(indice_4)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_5)] +  " (" +  "X_" + str(indice_5)  +"), \n" + \
                                        X_meaning_dictionnary_["X_" + str(indice_6)] + " (" +  "X_" + str(indice_6)  +"), "+ \
                                        X_meaning_dictionnary_["X_" + str(indice_7)] +  " (" +  "X_" + str(indice_7)  +"), \n" + \
                                         X_meaning_dictionnary_["X_" + str(indice_8)]   +  " (" +  "X_" + str(indice_8)  +"). \n"
    
    if workstep == "plotting_graphs_for_the_paper" or workstep == "computing_static_dynamic_score_for_paper" or  workstep == "finding_best_input_dataset_size":
        picture_title =  "Interactions between " + decapitalize( X_meaning_dictionnary_["X_" + str(cibled_indice)] ) + " (" +  "X_" + str(cibled_indice)  +")" +  " and others covariates, except little socket frequency (freq.) level."
        #          " Foreach plot, Pointwise marginal effects with respect to  " +  "X_" + str(cibled_indice)  +" (X axis), and relative to the secondary covariate (Y axis)"  
        fig.suptitle(picture_title, fontsize = paper_fontsize)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
       
    else : 
        plt.yticks(fontsize=8)
        fig.suptitle(picture_title)

        
   
    
    #plt.gcf().autofmt_xdate()
    
    if not repeat_experiments:
        plt.savefig(marginal_effect_exploration_folder_ + "/"+ picture_name)#, format="png", bbox_inches="tight")
        plt.savefig(marginal_effect_exploration_folder_ + "/"+ reduced_picture_name)#, format="png", bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()
    if   workstep == "plotting_graphs_for_the_paper" or workstep == "computing_static_dynamic_score_for_paper" or  workstep == "finding_best_input_dataset_size":
        print(" J-L mapping where J = " + str(cibled_indice) , repr(J_L_mapping))
        print("paper font size  = " + str(paper_fontsize) )
        return J_L_mapping


def plot_ten_marginal_interactions (X_train, pointwise_margins, cibled_indice, indice_0, indice_1, indice_2,  indice_3, indice_4, indice_5, indice_6, indice_7, indice_8 , indice_9, X_meaning_dictionnary_, marginal_effect_exploration_folder_, acceptable_marginal_mean_value = 0):

    print("--- In function plot_marginal_interactions : plotting d_X_" + str(cibled_indice) + " with regard to X_"  + str(indice_0) + ", X_" + str(indice_1) + ", X_" + str(indice_2) + ", ... X_" + str(indice_9))
    fig, ((d_X_cibled_indice_over_X_indice_0, d_X_cibled_indice_over_X_indice_1, d_X_cibled_indice_over_X_indice_2, d_X_cibled_indice_over_X_indice_3, d_X_cibled_indice_over_X_indice_4 ) , 
          ( d_X_cibled_indice_over_X_indice_5, d_X_cibled_indice_over_X_indice_6, d_X_cibled_indice_over_X_indice_7, d_X_cibled_indice_over_X_indice_8, d_X_cibled_indice_over_X_indice_9 )) = plt.subplots(nrows= 2, ncols = 5,  figsize=(27, 14))        
          

    transparency = 0.09


    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_0, X_train, pointwise_margins, indice_0, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)

    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_1, X_train, pointwise_margins, indice_1, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)

    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_2, X_train, pointwise_margins, indice_2, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)
    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_3, X_train, pointwise_margins, indice_3, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)
    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_4, X_train, pointwise_margins, indice_4, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)
    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_5, X_train, pointwise_margins, indice_5, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)
    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_6, X_train, pointwise_margins, indice_6, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)
    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_7, X_train, pointwise_margins, indice_7, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)
    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_8, X_train, pointwise_margins, indice_8, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)
    plot_single_marginal_plot(d_X_cibled_indice_over_X_indice_9, X_train, pointwise_margins, indice_9, cibled_indice, X_meaning_dictionnary_, transparency, acceptable_marginal_mean_value)

    
  

    #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")
    picture_name =  X_meaning_dictionnary_["X_" + str(cibled_indice)].replace(" ","_")[-16:] + \
                                              "_over_" +  X_meaning_dictionnary_["X_" + str(indice_0)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_2)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_3)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_4)].replace(" ","_")[-16:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_5)].replace(" ","_")[-16:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_6)].replace(" ","_")[-16:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_7)].replace(" ","_")[-16:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_8)].replace(" ","_")[-16:] + "__" + \
                                         X_meaning_dictionnary_["X_" + str(indice_9)].replace(" ","_")[-16:]  + ".pdf" 

    reduced_picture_name =  "X_" + str(cibled_indice) + \
                                              "_over_" + "X_" + str(indice_0) + "__" + \
                                           "X_" + str(indice_1)  + "__" + \
                                            "X_" + str(indice_2) + "__" + \
                                            "X_" + str(indice_3) + "__" + \
                                            "X_" + str(indice_4) + "__" + \
                                            "X_" + str(indice_5) + "__" + \
                                             "X_" + str(indice_6) + "__" + \
                                              "X_" + str(indice_7) + "__" + \
                                              "X_" + str(indice_8) + "__" + \
                                             "X_" + str(indice_9)  + ".pdf" 

    picture_title =  X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"\
                                              " over " +  X_meaning_dictionnary_["X_" + str(indice_0)] +   " (" +  "X_" + str(indice_0)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)] + " (" +  "X_" + str(indice_1)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_2)] + " (" +  "X_" + str(indice_2)  +"), \n"  + \
                                            X_meaning_dictionnary_["X_" + str(indice_3)] +  " (" +  "X_" + str(indice_3)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_4)] +  " (" +  "X_" + str(indice_4)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_5)] +  " (" +  "X_" + str(indice_5)  +"), \n" + \
                                        X_meaning_dictionnary_["X_" + str(indice_6)] + " (" +  "X_" + str(indice_6)  +"), "+ \
                                        X_meaning_dictionnary_["X_" + str(indice_7)] +  " (" +  "X_" + str(indice_7)  +"), \n" + \
                                        X_meaning_dictionnary_["X_" + str(indice_8)] +  " (" +  "X_" + str(indice_8)  +"), \n" + \
                                         X_meaning_dictionnary_["X_" + str(indice_9)]   +  " (" +  "X_" + str(indice_9)  +"). \n"

    fig.suptitle(picture_title)
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=8)
    plt.savefig(marginal_effect_exploration_folder_ + "/"+ picture_name)
    plt.savefig(marginal_effect_exploration_folder_ + "/"+ reduced_picture_name)
    plt.clf()
    plt.cla()
    plt.close()



def plot_twelve_marginal_interactions (X_train, pointwise_margins, cibled_indice, indice_0, indice_1, indice_2,  indice_3, indice_4, indice_5, indice_6, indice_7, indice_8 , indice_9, indice_10, indice_11, X_meaning_dictionnary_, marginal_effect_exploration_folder_):

    print("--- In function plot_marginal_interactions : plotting d_X_" + str(cibled_indice) + " with regard to X_"  + str(indice_0) + ", X_" + str(indice_1) + ", X_" + str(indice_2) + " ... X_" + str(indice_11))
    fig, ((d_X_cibled_indice_over_X_indice_0, d_X_cibled_indice_over_X_indice_1, d_X_cibled_indice_over_X_indice_2, d_X_cibled_indice_over_X_indice_3),
         (d_X_cibled_indice_over_X_indice_4, d_X_cibled_indice_over_X_indice_5, d_X_cibled_indice_over_X_indice_6, d_X_cibled_indice_over_X_indice_7),
          ( d_X_cibled_indice_over_X_indice_8, d_X_cibled_indice_over_X_indice_9, d_X_cibled_indice_over_X_indice_10, d_X_cibled_indice_over_X_indice_11)) = plt.subplots(nrows= 3, ncols = 4,  figsize=(24, 21))        
          

    transparency = 0.07
    # special trick to print acceptable plots
    d_X_cibled_indice_over_X_indice_0.scatter(X_train[:,indice_0], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_0.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_0]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_0.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_0)] +   " (" +  "X_" + str(indice_0)  +")")
    d_X_cibled_indice_over_X_indice_0.set_xlabel("X_" + str(indice_0) + ": " + X_meaning_dictionnary_["X_"+ str(indice_0)])
    #d_X_cibled_indice_over_X_indice_0.set_ylabel()
    d_X_cibled_indice_over_X_indice_0.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_0.set_xlim(xmin=-0.25)



    d_X_cibled_indice_over_X_indice_1.scatter(X_train[:,indice_1], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_1.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_1]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_1.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  + ")" + " over \n " +\
                                                    X_meaning_dictionnary_["X_" + str(indice_1)] +   " (" +  "X_" + str(indice_1)  +").")
    d_X_cibled_indice_over_X_indice_1.set_xlabel("X_" + str(indice_1) + ": " + X_meaning_dictionnary_["X_" + str(indice_1)])
    d_X_cibled_indice_over_X_indice_1.set_ylabel("d_X_" + str(cibled_indice) + " : pointwise marginal effect of " +  X_meaning_dictionnary_["X_" + str(cibled_indice)] )
    d_X_cibled_indice_over_X_indice_1.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_1.set_xlim(xmin=-0.25)



    d_X_cibled_indice_over_X_indice_2.scatter(X_train[:,indice_2], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_2.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_2]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_2.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_2)] +   " (" +  "X_" + str(indice_2)  +")")
    d_X_cibled_indice_over_X_indice_2.set_xlabel("X_" + str(indice_2) + ": " + X_meaning_dictionnary_["X_" + str(indice_2)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_2.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_2.set_xlim(xmin=-0.25)



    d_X_cibled_indice_over_X_indice_3.scatter(X_train[:,indice_3], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_3.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_3]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_3.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_3)] +   " (" +  "X_" + str(indice_3)  +")")
    d_X_cibled_indice_over_X_indice_3.set_xlabel("X_" + str(indice_3) + ": " + X_meaning_dictionnary_["X_" + str(indice_3)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_3.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_3.set_xlim(xmin=-0.25)


    ###################

    d_X_cibled_indice_over_X_indice_4.scatter(X_train[:,indice_4], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_4.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_4]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_4.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_4)] +   " (" +  "X_" + str(indice_4)  +")")
    d_X_cibled_indice_over_X_indice_4.set_xlabel("X_" + str(indice_3) + ": " + X_meaning_dictionnary_["X_" + str(indice_4)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_4.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_4.set_xlim(xmin=-0.25)



    d_X_cibled_indice_over_X_indice_5.scatter(X_train[:,indice_5], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_5.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_5]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_5.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_5)] +   " (" +  "X_" + str(indice_5)  +")")
    d_X_cibled_indice_over_X_indice_5.set_xlabel("X_" + str(indice_5) + ": " + X_meaning_dictionnary_["X_" + str(indice_5)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_5.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_5.set_xlim(xmin=-0.25)


    d_X_cibled_indice_over_X_indice_6.scatter(X_train[:,indice_6], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_6.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_6]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_6.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_6)] +   " (" +  "X_" + str(indice_6)  +")")
    d_X_cibled_indice_over_X_indice_6.set_xlabel("X_" + str(indice_6) + ": " + X_meaning_dictionnary_["X_" + str(indice_6)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_6.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_6.set_xlim(xmin=-0.25)


    d_X_cibled_indice_over_X_indice_7.scatter(X_train[:,indice_7], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_7.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_7]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_7.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_7)] +   " (" +  "X_" + str(indice_7)  +")")
    d_X_cibled_indice_over_X_indice_7.set_xlabel("X_" + str(indice_7) + ": " + X_meaning_dictionnary_["X_" + str(indice_7)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_7.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_7.set_xlim(xmin=-0.25)

    d_X_cibled_indice_over_X_indice_8.scatter(X_train[:,indice_8], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_8.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_8]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_8.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_8)] +   " (" +  "X_" + str(indice_8)  +")")
    d_X_cibled_indice_over_X_indice_8.set_xlabel("X_" + str(indice_8) + ": " + X_meaning_dictionnary_["X_" + str(indice_8)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_8.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_8.set_xlim(xmin=-0.25)


    d_X_cibled_indice_over_X_indice_9.scatter(X_train[:,indice_9], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_9.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_9]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_9.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_9)] +   " (" +  "X_" + str(indice_9)  +")")
    d_X_cibled_indice_over_X_indice_9.set_xlabel("X_" + str(indice_9) + ": " + X_meaning_dictionnary_["X_" + str(indice_9)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_9.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_9.set_xlim(xmin=-0.25)


    d_X_cibled_indice_over_X_indice_10.scatter(X_train[:,indice_10], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_10.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_10]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_10.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_10)] +   " (" +  "X_" + str(indice_10)  +")")
    d_X_cibled_indice_over_X_indice_10.set_xlabel("X_" + str(indice_10) + ": " + X_meaning_dictionnary_["X_" + str(indice_10)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_10.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_10.set_xlim(xmin=-0.25)


    d_X_cibled_indice_over_X_indice_11.scatter(X_train[:,indice_11], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_11.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_11]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_11.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_11)] +   " (" +  "X_" + str(indice_11)  +")")
    d_X_cibled_indice_over_X_indice_11.set_xlabel("X_" + str(indice_11) + ": " + X_meaning_dictionnary_["X_" + str(indice_11)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_11.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_11.set_xlim(xmin=-0.25)


    #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")
    picture_name =  X_meaning_dictionnary_["X_" + str(cibled_indice)].replace(" ","_")[-16:] + \
                                              "_over_" +  X_meaning_dictionnary_["X_" + str(indice_0)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_2)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_3)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_4)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_5)].replace(" ","_")[-14:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_6)].replace(" ","_")[-14:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_7)].replace(" ","_")[-14:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_8)].replace(" ","_")[-14:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_9)].replace(" ","_")[-14:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_10)].replace(" ","_")[-14:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_11)].replace(" ","_")[-14:]  + ".png" 

    reduced_picture_name =  "X_" + str(cibled_indice) + \
                                              "_over_" + "X_" + str(indice_0) + "__" + \
                                           "X_" + str(indice_1)  + "__" + \
                                            "X_" + str(indice_2) + "__" + \
                                            "X_" + str(indice_3) + "__" + \
                                            "X_" + str(indice_4) + "__" + \
                                            "X_" + str(indice_5) + "__" + \
                                             "X_" + str(indice_6) + "__" + \
                                              "X_" + str(indice_7) + "__" + \
                                              "X_" + str(indice_8) + "__" + \
                                               "X_" + str(indice_9) + "__" + \
                                                "X_" + str(indice_10) + "__" + \
                                             "X_" + str(indice_11)  + ".png" 

    picture_title =  X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"\
                                              " over " +  X_meaning_dictionnary_["X_" + str(indice_0)] +   " (" +  "X_" + str(indice_0)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)] + " (" +  "X_" + str(indice_1)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_2)] + " (" +  "X_" + str(indice_2)  +"), \n"  + \
                                            X_meaning_dictionnary_["X_" + str(indice_3)] +  " (" +  "X_" + str(indice_3)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_4)] +  " (" +  "X_" + str(indice_4)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_5)] +  " (" +  "X_" + str(indice_5)  +"), \n" + \
                                        X_meaning_dictionnary_["X_" + str(indice_6)] + " (" +  "X_" + str(indice_6)  +"), "+ \
                                        X_meaning_dictionnary_["X_" + str(indice_7)] +  " (" +  "X_" + str(indice_7)  +"), \n" + \
                                        X_meaning_dictionnary_["X_" + str(indice_8)] +  " (" +  "X_" + str(indice_8)  +"), \n" + \
                                         X_meaning_dictionnary_["X_" + str(indice_9)] +  " (" +  "X_" + str(indice_9)  +"), \n" + \
                                          X_meaning_dictionnary_["X_" + str(indice_10)] +  " (" +  "X_" + str(indice_10)  +"), \n" + \
                                         X_meaning_dictionnary_["X_" + str(indice_11)]   +  " (" +  "X_" + str(indice_11)  +"). \n"

    fig.suptitle(picture_title)
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=8)
    plt.savefig(marginal_effect_exploration_folder_ + "/"+ picture_name )
    plt.savefig(marginal_effect_exploration_folder_ + "/"+ reduced_picture_name )
    plt.clf()
    plt.cla()
    plt.close()






def plot_seven_marginal_interactions (X_train, pointwise_margins, cibled_indice, indice_0, indice_1, indice_2,  indice_3, indice_4, indice_5, indice_6,  X_meaning_dictionnary_, marginal_effect_exploration_folder_):

    print("--- In function plot_marginal_interactions : plotting d_X_" + str(cibled_indice) + " with regard to X_"  + str(indice_0) + ", X_" + str(indice_1) + ", X_" + str(indice_2) + " ... X_" + str(indice_6))
    fig, ((d_X_cibled_indice_over_X_indice_0, d_X_cibled_indice_over_X_indice_1, d_X_cibled_indice_over_X_indice_2),
         (d_X_cibled_indice_over_X_indice_3, d_X_cibled_indice_over_X_indice_4, nothing),
          ( d_X_cibled_indice_over_X_indice_5, d_X_cibled_indice_over_X_indice_6, nothing)) = plt.subplots(nrows= 3, ncols = 3,  figsize=(24, 21))        
          

    transparency = 0.07
    # special trick to print acceptable plots
    d_X_cibled_indice_over_X_indice_0.scatter(X_train[:,indice_0], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_0.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_0]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_0.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_0)] +   " (" +  "X_" + str(indice_0)  +")")
    d_X_cibled_indice_over_X_indice_0.set_xlabel("X_" + str(indice_0) + ": " + X_meaning_dictionnary_["X_"+ str(indice_0)])
    #d_X_cibled_indice_over_X_indice_0.set_ylabel()
    d_X_cibled_indice_over_X_indice_0.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_0.set_xlim(xmin=-0.25)



    d_X_cibled_indice_over_X_indice_1.scatter(X_train[:,indice_1], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_1.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_1]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_1.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  + ")" + " over \n " +\
                                                    X_meaning_dictionnary_["X_" + str(indice_1)] +   " (" +  "X_" + str(indice_1)  +").")
    d_X_cibled_indice_over_X_indice_1.set_xlabel("X_" + str(indice_1) + ": " + X_meaning_dictionnary_["X_" + str(indice_1)])
    d_X_cibled_indice_over_X_indice_1.set_ylabel("d_X_" + str(cibled_indice) + " : pointwise marginal effect of " +  X_meaning_dictionnary_["X_" + str(cibled_indice)] )
    d_X_cibled_indice_over_X_indice_1.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_1.set_xlim(xmin=-0.25)



    d_X_cibled_indice_over_X_indice_2.scatter(X_train[:,indice_2], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_2.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_2]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_2.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_2)] +   " (" +  "X_" + str(indice_2)  +")")
    d_X_cibled_indice_over_X_indice_2.set_xlabel("X_" + str(indice_2) + ": " + X_meaning_dictionnary_["X_" + str(indice_2)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_2.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_2.set_xlim(xmin=-0.25)



    d_X_cibled_indice_over_X_indice_3.scatter(X_train[:,indice_3], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_3.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_3]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_3.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_3)] +   " (" +  "X_" + str(indice_3)  +")")
    d_X_cibled_indice_over_X_indice_3.set_xlabel("X_" + str(indice_3) + ": " + X_meaning_dictionnary_["X_" + str(indice_3)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_3.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_3.set_xlim(xmin=-0.25)


    ###################

    d_X_cibled_indice_over_X_indice_4.scatter(X_train[:,indice_4], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_4.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_4]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_4.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_4)] +   " (" +  "X_" + str(indice_4)  +")")
    d_X_cibled_indice_over_X_indice_4.set_xlabel("X_" + str(indice_3) + ": " + X_meaning_dictionnary_["X_" + str(indice_4)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_4.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_4.set_xlim(xmin=-0.25)



    d_X_cibled_indice_over_X_indice_5.scatter(X_train[:,indice_5], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_5.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_5]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_5.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_5)] +   " (" +  "X_" + str(indice_5)  +")")
    d_X_cibled_indice_over_X_indice_5.set_xlabel("X_" + str(indice_5) + ": " + X_meaning_dictionnary_["X_" + str(indice_5)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_5.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_5.set_xlim(xmin=-0.25)


    d_X_cibled_indice_over_X_indice_6.scatter(X_train[:,indice_6], pointwise_margins[:,cibled_indice], s=2000 , alpha=transparency,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_6.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_6]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_6.set_title( X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"+ \
                                                           " over \n " + X_meaning_dictionnary_["X_" + str(indice_6)] +   " (" +  "X_" + str(indice_6)  +")")
    d_X_cibled_indice_over_X_indice_6.set_xlabel("X_" + str(indice_6) + ": " + X_meaning_dictionnary_["X_" + str(indice_6)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_6.tick_params(size=8)
    d_X_cibled_indice_over_X_indice_6.set_xlim(xmin=-0.25)


    #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")
    picture_name =  X_meaning_dictionnary_["X_" + str(cibled_indice)].replace(" ","_")[-16:] + \
                                              "_over_" +  X_meaning_dictionnary_["X_" + str(indice_0)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_2)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_3)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_4)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_5)].replace(" ","_")[-14:] + "__" + \
                                        X_meaning_dictionnary_["X_" + str(indice_6)].replace(" ","_")[-14:]  + ".png" 

    reduced_picture_name =  "X_" + str(cibled_indice) + \
                                              "_over_" + "X_" + str(indice_0) + "__" + \
                                           "X_" + str(indice_1)  + "__" + \
                                            "X_" + str(indice_2) + "__" + \
                                            "X_" + str(indice_3) + "__" + \
                                            "X_" + str(indice_4) + "__" + \
                                            "X_" + str(indice_5) + "__" + \
                                             "X_" + str(indice_6)  + ".png" 

    picture_title =  X_meaning_dictionnary_["X_" + str(cibled_indice)] + " (" +  "X_" + str(cibled_indice)  +")"\
                                              " over " +  X_meaning_dictionnary_["X_" + str(indice_0)] +   " (" +  "X_" + str(indice_0)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)] + " (" +  "X_" + str(indice_1)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_2)] + " (" +  "X_" + str(indice_2)  +"), \n"  + \
                                            X_meaning_dictionnary_["X_" + str(indice_3)] +  " (" +  "X_" + str(indice_3)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_4)] +  " (" +  "X_" + str(indice_4)  +")," + \
                                            X_meaning_dictionnary_["X_" + str(indice_5)] +  " (" +  "X_" + str(indice_5)  +"), \n" + \
                                         X_meaning_dictionnary_["X_" + str(indice_6)]   +  " (" +  "X_" + str(indice_6)  +"). \n"

    fig.suptitle(picture_title)
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=8)
    plt.savefig(marginal_effect_exploration_folder_ + "/"+ picture_name )
    plt.savefig(marginal_effect_exploration_folder_ + "/"+ reduced_picture_name )
    plt.clf()
    plt.cla()
    plt.close()



"""
def plot_marginal_interactions (X_train, pointwise_margins, cibled_indice, indice_0, indice_1, indice_2,  X_meaning_dictionnary_, marginal_effect_exploration_folder_ ):

    print("--- In function plot_marginal_interactions : plotting d_X_" + str(cibled_indice) + " with regard to X_"  + str(indice_0) + ", X_" + str(indice_1) + ", X_" + str(indice_2))
    _, ((d_X_cibled_indice_over_X_indice_0, d_X_cibled_indice_over_X_indice_1) ,
               (d_X_cibled_indice_over_X_indice_2, d_X_cibled_indice_over_X_indice_3),
               (d_X_cibled_indice_over_X_indice_4, d_X_cibled_indice_over_X_indice_5),
               (d_X_cibled_indice_over_X_indice_6, d_X_cibled_indice_over_X_indice_7)) = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(6, 13))

    # special trick to print acceptable plots
    d_X_cibled_indice_over_X_indice_0.scatter(X_train[:,indice_0], pointwise_margins[:,cibled_indice], s=2000 , alpha=0.01,  c = "blue")
    for current_line_index in range(0, 40):
        d_X_cibled_indice_over_X_indice_0.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_0]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
   

    # Add title and axis names
    d_X_cibled_indice_over_X_indice_0.set_title("d_X_"+ str(cibled_indice) +" over X_"+ str(indice_0))
    d_X_cibled_indice_over_X_indice_0.set_xlabel("X_" + str(indice_0) + ": " + X_meaning_dictionnary_["X_"+ str(indice_0)])
    #d_X_cibled_indice_over_X_indice_0.set_ylabel()
    d_X_cibled_indice_over_X_indice_0.tick_params(size=8)

    #d_X_cibled_indice_over_X_indice_1.scatter(X_train[:,indice_1], pointwise_margins[:,cibled_indice],  s=180, alpha=0.1, c = "blue")
    for current_line_index in range(0, 15):
        d_X_cibled_indice_over_X_indice_1.scatter([var+0.005*current_line_index for var in X_train[:,indice_1]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_1.scatter([var+0.15 for var in X_train[:,indice_1]], pointwise_margins[:,cibled_indice], s=360 , alpha=0.01,  c = "blue")


    # Add title and axis names
    d_X_cibled_indice_over_X_indice_1.set_title("d_X_"+ str(cibled_indice) + " over X_" + str(indice_1))
    d_X_cibled_indice_over_X_indice_1.set_xlabel("X_" + str(indice_1) + ": " + X_meaning_dictionnary_["X_" + str(indice_1)])
    d_X_cibled_indice_over_X_indice_1.set_ylabel("d_X_" + str(cibled_indice) + " : pointwise marginal effect of " +  X_meaning_dictionnary_["X_" + str(cibled_indice)] )
    d_X_cibled_indice_over_X_indice_1.tick_params(size=8)


    #d_X_cibled_indice_over_X_indice_2.scatter(X_train[:,indice_2], pointwise_margins[:,cibled_indice],  s=2, marker = "_" , c = "blue")

    for current_line_index in range(0, 15):
        d_X_cibled_indice_over_X_indice_2.scatter([var+0.005*current_line_index for var in X_train[:,indice_2]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_2.scatter([var+0.15 for var in X_train[:,indice_2]], pointwise_margins[:,cibled_indice], s=360 , alpha=0.01,  c = "blue")

    # Add title and axis names
    d_X_cibled_indice_over_X_indice_2.set_title("d_X_" + str(cibled_indice) +" over X_" + str(indice_2) )
    d_X_cibled_indice_over_X_indice_2.set_xlabel("X_" + str(indice_2) + ": " + X_meaning_dictionnary_["X_" + str(indice_2)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_2.tick_params(size=8)


    #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")
    picture_name =  X_meaning_dictionnary_["X_" + str(cibled_indice)].replace(" ","_")[-14:] + \
                                              "_over_" +  X_meaning_dictionnary_["X_" + str(indice_0)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)].replace(" ","_")[-14:] + "__" + \
                                         X_meaning_dictionnary_["X_" + str(indice_2)].replace(" ","_")[-14:]  + ".png" 
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=8)
    plt.savefig(marginal_effect_exploration_folder_ + "/"+ picture_name )

    plt.clf()
    plt.cla()
    plt.close()
"""



def plot_marginal_interactions_old_version (X_train, pointwise_margins, cibled_indice, indice_0, indice_1, indice_2,  X_meaning_dictionnary_, marginal_effect_exploration_folder_ ):

    print("--- In function plot_marginal_interactions : plotting d_X_" + str(cibled_indice) + " with regard to X_"  + str(indice_0) + ", X_" + str(indice_1) + ", X_" + str(indice_2))
    _, (d_X_cibled_indice_over_X_indice_0, d_X_cibled_indice_over_X_indice_1, d_X_cibled_indice_over_X_indice_2) = plt.subplots(nrows= 3, sharex=True, sharey=True, figsize=(6, 13))

    # special trick to print acceptable plots
    d_X_cibled_indice_over_X_indice_0.scatter(X_train[:,indice_0], pointwise_margins[:,cibled_indice], s=2000 , alpha=0.01,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_0.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_0]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
    d_X_cibled_indice_over_X_indice_0.set_xlim(xmin=-0.25)

    # Add title and axis names
    d_X_cibled_indice_over_X_indice_0.set_title("d_X_"+ str(cibled_indice) +" over X_"+ str(indice_0))
    d_X_cibled_indice_over_X_indice_0.set_xlabel("X_" + str(indice_0) + ": " + X_meaning_dictionnary_["X_"+ str(indice_0)])
    #d_X_cibled_indice_over_X_indice_0.set_ylabel()
    d_X_cibled_indice_over_X_indice_0.tick_params(size=8)

    #d_X_cibled_indice_over_X_indice_1.scatter(X_train[:,indice_1], pointwise_margins[:,cibled_indice],  s=180, alpha=0.1, c = "blue")




    d_X_cibled_indice_over_X_indice_1.scatter(X_train[:,indice_1], pointwise_margins[:,cibled_indice], s=2000 , alpha=0.01,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_1.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_1]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
   
    # Add title and axis names
    d_X_cibled_indice_over_X_indice_1.set_title("d_X_"+ str(cibled_indice) + " over X_" + str(indice_1))
    d_X_cibled_indice_over_X_indice_1.set_xlabel("X_" + str(indice_1) + ": " + X_meaning_dictionnary_["X_" + str(indice_1)])
    d_X_cibled_indice_over_X_indice_1.set_ylabel("d_X_" + str(cibled_indice) + " : pointwise marginal effect of " +  X_meaning_dictionnary_["X_" + str(cibled_indice)] )
    d_X_cibled_indice_over_X_indice_1.tick_params(size=8)


    #d_X_cibled_indice_over_X_indice_2.scatter(X_train[:,indice_2], pointwise_margins[:,cibled_indice],  s=2, marker = "_" , c = "blue")

    d_X_cibled_indice_over_X_indice_2.scatter(X_train[:,indice_2], pointwise_margins[:,cibled_indice], s=2000 , alpha=0.01,  c = "blue")
    for current_line_index in range(0, 50):
        d_X_cibled_indice_over_X_indice_2.scatter([var + 0.3 + 0.005*current_line_index for var in X_train[:,indice_2]], pointwise_margins[:,cibled_indice], s=10, marker = "_",  c = "blue")
   
    # Add title and axis names
    d_X_cibled_indice_over_X_indice_2.set_title("d_X_" + str(cibled_indice) +" over X_" + str(indice_2) )
    d_X_cibled_indice_over_X_indice_2.set_xlabel("X_" + str(indice_2) + ": " + X_meaning_dictionnary_["X_" + str(indice_2)])
    #d_X_cibled_indice_over_X_indice_2.set_ylabel(    )
    d_X_cibled_indice_over_X_indice_2.tick_params(size=8)


    #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")
    picture_name =  X_meaning_dictionnary_["X_" + str(cibled_indice)].replace(" ","_")[-14:] + \
                                              "_over_" +  X_meaning_dictionnary_["X_" + str(indice_0)].replace(" ","_")[-14:] + "__" + \
                                            X_meaning_dictionnary_["X_" + str(indice_1)].replace(" ","_")[-14:] + "__" + \
                                         X_meaning_dictionnary_["X_" + str(indice_2)].replace(" ","_")[-14:]  + ".png" 
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=8)
    plt.savefig(marginal_effect_exploration_folder_ + "/"+ picture_name )

    plt.clf()
    plt.cla()
    plt.close()

