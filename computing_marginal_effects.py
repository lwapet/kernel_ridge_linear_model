#from types import NoneType
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import utils_functions as utils
import statsmodels.api as sm
import statsmodels.sandbox as sm_sandbox

import statsmodels
from statsmodels.sandbox.regression.kernridgeregress_class import GaussProcess

from scipy import spatial as ssp
import math 

# Those matrices are usefull to easyly make Sigma Som using the diag_of_product_functions

# This matrix, for each couple of observations (i, j) saves the value of || X [i, :] - X[j, :] ||
distance_matrix_raw_by_raw_on_X= []

# this matrix, for each couple of observations (i,j) , saves the value of  ci * exp (-|| X [i, :] - X[j, :] ||/ sigma_2)
ci_exp_matrix = []


# this matrix caches dif_matrix of dimension M*N*N,  and each j variable there it the matrix dif (i1, i2), difference between values of vector X on column J  
dif_matrix_cubic = []


# step 1, function to compute Dif (i,j) where Dif_i_j = vector (x - X_i_j, x in the column j of X)
def difference_relative_to(array, center):
    result = []
    #print("** in function  difference_relative_to")
    #print("array: " ,array)
    #print("center: ", center)
    for n in array:
        result.append(n-center)
    return result 



def get_dif_Vector(X, i, j):
    Dif = []
    N = len(X)
    # filling Dif on each row, after we will transpose
    result = [] # in the form of a column
    result_row_to_be_column = []
    """
    for i in range(0,N):
        result_row_to_be_column.append(difference_relative_to(X[:,j], X[i,j]))
    result = np.array(result_row_to_be_column).T  # juts to transpose it
    """
    result= np.array(difference_relative_to(X[:,j], X[i,j]))
    #print (" * End computing dif_vector_" + str(i)+"_"+str(j) + ", result computed with loop : ",result)
    return result
        

# ### step 2 computing dif_matrixj; it  is an N*N matrix where at each column n of dif_matrix, we have get_dif_Vector(X,n,j)
def dif_matrix(X,j):
    #print (" ** START : computing dif_matrix on variable " + str(j))

    global dif_matrix_cubic 
    if len(dif_matrix_cubic) <= j : # we  want to compute dif_matrix_cubic[j], we test if this value is not present
                                    # we test if it size is not j+1. (if we only have indices 0,...,j-1)
        #print ("  --> the dif_matrix for the variable " + str(j) + " is not yet computed")
        dif_matrix_T_j = []
        N = len(X)
        for i in range (0, N):
            dif_matrix_T_j.append(get_dif_Vector(X, i, j))
        dif_matrix_T_j = np.array(dif_matrix_T_j)
        #print (" ** END : first computing dif_matrix on variable " + str(j) + ", transposed result = " , dif_matrix_T_j)
        #print (" ** END : first computing dif_matrix on variable " + str(j) + ", result = " , dif_matrix_T_j.T)
        dif_matrix_cubic.append(dif_matrix_T_j.T)
        return dif_matrix_T_j.T

    else: 
        #print ("  --> the dif_matrix for the variable " + str(j) + " is already  computed ")
        #print (" ** END : retriving dif_matrix on variable " + str(j) + ", result = " , dif_matrix_cubic[j] )
        return dif_matrix_cubic[j] 
   

# step 3 For an observation i function to compute  exp_terms (i), exp_terms (i) = vector of n values, where each is  [exponential(- square_norm(X[n,:] , X[i:])/ sigma_2)]
def exponential_minus_sigma2(value_in_array, sigma_2):
    result = math.exp(-1*value_in_array/sigma_2)
    return result 


def only_exp_terms(X, i, sigma_2):
    result = []
    N = len(X)
    #print ("** START computing exp_terms on observation ", i)
    #print ("** X = ", X)
    global distance_matrix_raw_by_raw_on_X
    if len(distance_matrix_raw_by_raw_on_X) == 0:
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

#step 4: function to compute efficiently the diagonal of a product of tow vectors. 
def diag_of_product(A, B):
    #print ("** START computing diag_of_product of A and B  ")
    #print ("A = ", A)
    #print ("B = ", B)
    result = (np.einsum('ij,ji->i', A,B)).T  # einsum is a workaround obtained from stack overflow 
    #print ("** END computing diag_of_product of A and B , result = ", result)
    return result


# step 5: function to compute ci_exp_term(i) which is  diag_of_product( C_vector, exp_term(X, i) transposed )
def ci_dot_exp___big_term(X, i, c_vector, sigma_2):
    #print ("** START computing ci exp vector of observation  ", i)
    print ("X = ", X)
    #result = diag_of_product(c_vector, (only_exp_terms(X,i))[:,np.newaxis], sigma_2 ) 
    result = diag_of_product(c_vector, (only_exp_terms(X, i, sigma_2)).T) ##### This was the error !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:)
    #print ("** END computing ci exp vector of observation  " + str(i) + ", -result = ",(np.array(result)[:,np.newaxis]))
    return np.array(result)


# step 5 computing the vector ci_exp_matrix where ci_exp_matrix has N lines and at line n we have the transposed of the vector ci_exp_term(i)
# in the paper image where I did the demonstration, it is called Diag
def get_ci_exp_matrix(X, c_vector, sigma_2):
    #print ("*** START computing ci exp matrix ")
    print ("X = ", X)
    global ci_exp_matrix
    if len(ci_exp_matrix)== 0:
        ci_exp_matrix = []
        N = len(X)
        for i in range (0, N):
            ci_exp_matrix.append((ci_dot_exp___big_term(X, i, c_vector, sigma_2)))
        
        #print ("*** END computing ci exp matrix, first computation result ", np.array(ci_exp_matrix))
    #else:
        #print ("*** END computing ci exp matrix, cached  result ", np.array(ci_exp_matrix))
    
    return np.array(ci_exp_matrix)

#step 6 function to compute the marginal effect of variable j Y_prim_hat_at_var_j
def Y_prim_hat_at_var (X,j,sigma_2, ci_exp_matrix): 
    #print("*** START computing margin of variable X", j)
    N = len(X)
    result = (1/N) * (-2/sigma_2) * np.sum( diag_of_product(ci_exp_matrix, dif_matrix(X,j))   ) 
    #print("*** END  computing margin of variable X" + str(j) + ", ---------------- result = " + str(result))
    return result

#step 6_B function to compute the pointwise marginal effect of variable j  at each observations
def Y_prim_hat_at_var_at_each_observations (X,j,sigma_2, ci_exp_matrix): 
    #print("*** START computing margin of variable X", j)
    N = len(X)
    result = np.multiply( diag_of_product(ci_exp_matrix, dif_matrix(X,j)), (-2/sigma_2) )  # return a vector, at each observation i, there is the product of the ith line 
                                                               # of ci_exp_matrix and the ith colum of dif_matrix(X, j)
                                                               # and at each observation these value computed is the point wise marginal effect
    #print("*** END  computing margin of variable X" + str(j) + ", ---------------- result = " + str(result))
    return result


# step 7 fonction to compute the marginal effect vector 
# WARNING the c_vector should be of dimension (M,1) where M is the number of variables. 
def marginal_effect(X, c_vector, sigma_2, repeat_experiments = False): 
    if repeat_experiments:
        # This matrix, for each couple of observations (i, j) saves the value of || X [i, :] - X[j, :] ||
        global distance_matrix_raw_by_raw_on_X
        distance_matrix_raw_by_raw_on_X = []

        # this matrix, for each couple of observations (i,j) , saves the value of  ci * exp (-|| X [i, :] - X[j, :] ||/ sigma_2)
        global ci_exp_matrix
        ci_exp_matrix = []


        # this matrix caches dif_matrix of dimension M*N*N,  and each j variable there it the matrix dif (i1, i2), difference between values of vector X on column J  
        global dif_matrix_cubic
        dif_matrix_cubic = []



    print(" ***** START in function marginal_effect *****")
    print ("X = ", X)
    margins = []
    pointwise_margin = []
    
    M = len(X [0])
    print ("number of variables: ",M)
    for j in range (0,M):
        margins.append(Y_prim_hat_at_var(X, j , sigma_2, get_ci_exp_matrix(X, c_vector, sigma_2)))
        pointwise_margin.append(Y_prim_hat_at_var_at_each_observations(X, j , sigma_2, get_ci_exp_matrix(X, c_vector, sigma_2)))
    
    pointwise_margin = np.array(pointwise_margin)
    print(" ***** END in  marginal_effect *****, margin = ", margins)
    print(" ***** END in function  marginal_effect *****, pointwise margin = ", pointwise_margin.T)
    return pointwise_margin.T, margins  

"""
# step 7 fonction to compute the ointwise marginal effect matrix 
# WARNING the c_vector should be of dimension (M,1) where M is the number of variables. 
def marginal_effect(X, c_vector, sigma_2): 
    print(" ***** START in function pointwise marginal_effect *****")
    print ("X = ", X)
    pointwise_margin = []
    
    M = len(X [0])
    print ("number of variables: ",M)
    for j in range (0,M):
        pointwise_margin.append(Y_prim_hat_at_var_at_each_observations(X, j , sigma_2, get_ci_exp_matrix(X, c_vector, sigma_2)))
    pointwise_margin = np.array(pointwise_margin)
    print(" ***** END in function pointwise marginal_effect *****, result = ", pointwise_margin.T)
    return pointwise_margin.T
"""




# step 8, naive implementation
def naive_marginal_effect(X, c_vector, sigma_2): 
    print(" ***** START in function naive marginal_effect *****")
    print ("X = ", X)
    margins = []
    pointwise_margins = []
    N = len(X)
    M = len(X [0])
    naive_y_j = 0
    print ("number of variables: ",M)
    print ("naive_y_j: ",naive_y_j)
    for j in range (0,M):
        naive_y_j = 0
        pointwise_margin = []
        for i in range (0,N):
            internal_sum_term=0
            #print ("internal sum term: ",internal_sum_term)
            for n in range (0,N):
                squared_norm_term = ( ssp.distance.euclidean(X[n,:], X[i,:]) )**2
                exp_term = math.exp(-1*squared_norm_term/sigma_2)
                ci_exp_big_term= c_vector[n,0] * exp_term
                diff_term = X[n,j] - X[i,j]
                #print("diff_term", diff_term)
                #print("ci term ", c_vector[n,0])
                #print("ci_exp_only_term ", exp_term)
                #print("ci_exp_big_term ", ci_exp_big_term)
                internal_sum_term =  internal_sum_term + ci_exp_big_term*diff_term 
                  
            #print ("internal sum term: ",internal_sum_term)
            pointwise_margin.append((-2/sigma_2) * internal_sum_term)
            naive_y_j =  naive_y_j + internal_sum_term 
        #print("appending value",  (1/N) * (-2/sigma_2) * naive_y_j ) 
        pointwise_margins.append(pointwise_margin)             
        margins.append(  (1/N) * (-2/sigma_2) *naive_y_j)

    pointwise_margins = np.array(pointwise_margins)
    print(" ***** END in function marginal_effect  with naïve approach*****,--------------------- margins = ", margins)
    print(" ***** END in function marginal_effect  with naïve approach*****,--------------------- pointwise margin = ", pointwise_margins.T)
    return  pointwise_margins.T, np.array(margins)
 