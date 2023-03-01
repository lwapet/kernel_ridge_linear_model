 
        
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

    # computing the marginal effect of the observation with naive implementation
    """
    print (" ***** START computing marginal effects with loop***** ")
    print ("X = ", X_train)
    n_pointwise_margins, n_margins = comput_margin.naive_marginal_effect(X_train, c_vector, sigma_2)
    print (" ***** END computing marginal effects ***** ")
    print("naive margins",  n_margins)
    print("margins",  margins)
    print("naive pointwise margins",  n_pointwise_margins)
    print("pointwise margins",  pointwise_margins)
    print("test of correctness means  = " + str(np.sum(n_pointwise_margins[:,0]) / len(X_train)) + 
        " direct value = ", n_margins[0]  )
    """


    # generating linear regression coefficients
    ols = sm.OLS(y_train, X_train)  # warning in the sm OLS function argument format, y is the first parameter. 
    reg = ols.fit()
    reg_pred_y_test = reg.predict(X_test)
    linear_coefficients = reg.params
    print("Predicted y test = ", reg_pred_y_test)
    print("linear model parameters  = ",  linear_coefficients)
    print("*** Linear model R2 score  = ", utils.compute_r2_score(y_test, reg_pred_y_test) )

    if phone_name == "samsung_galaxy_s8" :

        linear_coeff_vs_kernel_ridge_margins_file = marginal_effect_exploration_folder + "/linear_coeff_vs_kernel_ridge_margins.csv" # Can change depending on the r2 score

        X_meaning_dictionnary = base_Y__X_meaning_dictionnary if X_format_in_model == "base_Y"  else base_Y_N_on_socket__X_meaning_dictionnary if  X_format_in_model == "base_Y_N_on_socket" else {}                 

        #Capturing linear coefficients and kernel ridge means marginal effect (not pointwise) in a file
        utils.capture_kernel_means_marginal_and_linear_model_coeff(margins, linear_coefficients, linear_coeff_vs_kernel_ridge_margins_file, X_meaning_dictionnary)
        
        if  X_format_in_model == "base_Y":

            ### Plotting X_1 distribution plot (Note, it is the activation state of the first core! because we are in Base_Y format of X).
            # plotting histograph
        
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,0], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_little_socket_frequency_level.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,1], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_0_state.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,5], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_big_socket_frequency_level.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,8], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_7_state.png", 3e-11, 8)

            
            ### Plotting marginal effect plots
            ## Regression of d_X_5 over all other variable including X_5 is the frequency of big cores
            d_X_5_coefficients_file = marginal_effect_exploration_folder + "/d_X_5_linear_coefficients.csv"
            d_X_5_ols = sm.OLS(pointwise_margins[:,5], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_5_reg = d_X_5_ols.fit()
            d_X_5_linear_coefficients = d_X_5_reg.params
            print("d_X_5 linear model parameters  = ",  d_X_5_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 5, 
                        d_X_i_linear_coefficients = d_X_5_linear_coefficients, 
                            file_path = d_X_5_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_5 over other variables")

            utils.plot_ten_marginal_interactions(X_train, pointwise_margins, 5, 0, 1, 2, 3,4,5,6,7, 8, 9, X_meaning_dictionnary, marginal_effect_exploration_folder, acceptable_marginal_mean_value)
        
            # processing d_X_0
            d_X_0_coefficients_file = marginal_effect_exploration_folder + "/d_X_0_linear_coefficients.csv"
            d_X_0_ols = sm.OLS(pointwise_margins[:,0], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_0_reg = d_X_0_ols.fit()
            d_X_0_linear_coefficients = d_X_0_reg.params
            print("d_X_0 linear model parameters  = ",  d_X_0_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 0, 
                        d_X_i_linear_coefficients = d_X_0_linear_coefficients, 
                            file_path = d_X_0_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_0 over other variables")

            utils.plot_ten_marginal_interactions(X_train, pointwise_margins, 0, 0, 1, 2, 3,4,5,6,7, 8, 9, X_meaning_dictionnary, marginal_effect_exploration_folder, acceptable_marginal_mean_value)




            # processing d_X_1 (core 0 state)

            d_X_1_coefficients_file = marginal_effect_exploration_folder + "/d_X_1_linear_coefficients.csv"
            d_X_1_ols = sm.OLS(pointwise_margins[:,1], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_1_reg = d_X_1_ols.fit()
            d_X_1_linear_coefficients = d_X_1_reg.params
            print("d_X_1 linear model parameters  = ",  d_X_1_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 1, 
                        d_X_i_linear_coefficients = d_X_1_linear_coefficients, 
                            file_path = d_X_1_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_1 over other variables")

            utils.plot_ten_marginal_interactions(X_train, pointwise_margins, 1, 0, 1, 2, 3,4,5,6,7, 8, 9, X_meaning_dictionnary, marginal_effect_exploration_folder, acceptable_marginal_mean_value)


            # processing d_X_6 (core 6 state)

            d_X_6_coefficients_file = marginal_effect_exploration_folder + "/d_X_6_linear_coefficients.csv"
            d_X_6_ols = sm.OLS(pointwise_margins[:,6], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_6_reg = d_X_6_ols.fit()
            d_X_6_linear_coefficients = d_X_6_reg.params
            print("d_X_6 linear model parameters  = ",  d_X_6_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 6, 
                        d_X_i_linear_coefficients = d_X_6_linear_coefficients, 
                            file_path = d_X_6_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_6 over other variables")

            utils.plot_ten_marginal_interactions(X_train, pointwise_margins, 6, 0, 1, 2, 3,4,5,6,7, 8, 9, X_meaning_dictionnary, marginal_effect_exploration_folder, acceptable_marginal_mean_value)
           
           # processing d_X_4 (core 3 state)

            d_X_4_coefficients_file = marginal_effect_exploration_folder + "/d_X_4_linear_coefficients.csv"
            d_X_4_ols = sm.OLS(pointwise_margins[:,4], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_4_reg = d_X_4_ols.fit()
            d_X_4_linear_coefficients = d_X_4_reg.params
            print("d_X_4 linear model parameters  = ",  d_X_4_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 4, 
                        d_X_i_linear_coefficients = d_X_4_linear_coefficients, 
                            file_path = d_X_4_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_4 over other variables")

            utils.plot_ten_marginal_interactions(X_train, pointwise_margins, 4, 0, 1, 2, 3,4,5,6,7, 8, 9, X_meaning_dictionnary, marginal_effect_exploration_folder, acceptable_marginal_mean_value)


            # processing d_X_9 (core 7 state)
            d_X_9_coefficients_file = marginal_effect_exploration_folder + "/d_X_9_linear_coefficients.csv"
            d_X_9_ols = sm.OLS(pointwise_margins[:,9], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_9_reg = d_X_9_ols.fit()
            d_X_9_linear_coefficients = d_X_9_reg.params
            print("d_X_9 linear model parameters  = ",  d_X_9_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 9, 
                        d_X_i_linear_coefficients = d_X_9_linear_coefficients, 
                            file_path = d_X_9_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_9 over other variables")

            utils.plot_ten_marginal_interactions(X_train, pointwise_margins, 9, 0, 1, 2, 3,4,5,6,7, 8, 9, X_meaning_dictionnary, marginal_effect_exploration_folder, acceptable_marginal_mean_value)


        
            """

            ## Regression of d_X_7 over all other variable including 
            d_X_7_coefficients_file = marginal_effect_exploration_folder + "/d_X_7_linear_coefficients.csv"
            d_X_7_ols = sm.OLS(pointwise_margins[:,7], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_7_reg = d_X_7_ols.fit()
            d_X_7_linear_coefficients = d_X_7_reg.params
            print("X_7_d linear model parameters  = ",  d_X_7_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 7, 
                        d_X_i_linear_coefficients = d_X_7_linear_coefficients, 
                            file_path = d_X_7_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_7 over other variables")
            utils.plot_marginal_interactions(X_train, pointwise_margins, 7, 0, 1, 2, 3,4,5,6,7, 8, X_meaning_dictionnary, marginal_effect_exploration_folder)
            
        

            ## Regression of d_X_0 over all other variable including 
            d_X_0_coefficients_file = marginal_effect_exploration_folder + "/d_X_0_linear_coefficients.csv"
            d_X_0_ols = sm.OLS(pointwise_margins[:,0], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_0_reg = d_X_0_ols.fit()
            d_X_0_linear_coefficients = d_X_0_reg.params
            print("X_0_d linear model parameters  = ",  d_X_0_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 0, 
                        d_X_i_linear_coefficients = d_X_0_linear_coefficients, 
                            file_path = d_X_0_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_0 over other variables")
            utils.plot_marginal_interactions(X_train, pointwise_margins, 0, 0, 1, 2, 3,4,5,6,7, 8, X_meaning_dictionnary, marginal_effect_exploration_folder)
            
            
            
            ## Regression of d_X_0 over all other variable including 
            d_X_1_coefficients_file = marginal_effect_exploration_folder + "/d_X_0_linear_coefficients.csv"
            d_X_1_ols = sm.OLS(pointwise_margins[:,1], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_1_reg = d_X_1_ols.fit()
            d_X_1_linear_coefficients = d_X_1_reg.params
            print("X_0_d linear model parameters  = ",  d_X_1_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 1, 
                        d_X_i_linear_coefficients = d_X_1_linear_coefficients, 
                            file_path = d_X_1_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_0 over other variables")
            utils.plot_marginal_interactions(X_train, pointwise_margins, 1, 0, 1, 2, 3,4,5,6,7, 8, X_meaning_dictionnary, marginal_effect_exploration_folder)
            
            """
             
        elif X_format_in_model == "base_Y_N_on_socket":
              ### Plotting X_1 distribution plot (Note, it is the activation state of the first core! because we are in Base_Y format of X).
            # plotting histograph
        
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,0], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_little_socket_frequency_level.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,1], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_number_of_little_cores_actives.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,5], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_big_socket_frequency_level.png", 3e-11, 3)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,9], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_7_frequency_level.png", 3e-11, 4)

            
            d_X_2_coefficients_file = marginal_effect_exploration_folder + "/d_X_2_linear_coefficients.csv"
            d_X_2_ols = sm.OLS(pointwise_margins[:,2], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_2_reg = d_X_2_ols.fit()
            d_X_2_linear_coefficients = d_X_2_reg.params
            print("X_2_d linear model parameters  = ",  d_X_2_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 2, 
                        d_X_i_linear_coefficients = d_X_2_linear_coefficients, 
                            file_path = d_X_2_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
        
            
            # plotting of d_X_2, regarding to other_variables with 
            _, (d_X_2_over_X_0, d_X_2_over_X_1, d_X_2_over_X_3) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 13))
            d_X_2_over_X_0.scatter(X_train[:,0], pointwise_margins[:,2], c = "blue")
            # Add title and axis names
            d_X_2_over_X_0.set_title('d_X_2 over X_0')
            d_X_2_over_X_0.set_xlabel('X_0 : frequency level of little socket')
            d_X_2_over_X_0.set_ylabel("d_X_2 : pointwise marginal effect of frequency of Medium core")
            d_X_2_over_X_0.tick_params(size=8)


            d_X_2_over_X_1.scatter(X_train[:,1], pointwise_margins[:,2],  c = "blue")
            # Add title and axis names
            d_X_2_over_X_1.set_title('d_X_2 over X_1')
            d_X_2_over_X_1.set_xlabel('X_1 : Number of threads on little socket')
            d_X_2_over_X_1.set_ylabel("d_X_2 ")
            d_X_2_over_X_1.tick_params(size=8)

        
            d_X_2_over_X_3.scatter(X_train[:,3], pointwise_margins[:,2],  c = "blue")
            # Add title and axis names
            d_X_2_over_X_3.set_title('d_X_2 over X_3')
            d_X_2_over_X_3.set_xlabel('X_3 : frequency of core 7 (8th core)')
            d_X_2_over_X_3.set_ylabel("d_X_2 : pointwise marginal effect of frequency of Medium core")
            d_X_2_over_X_3.tick_params(size=8)

            #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")

            plt.gcf().autofmt_xdate()
            plt.xticks(fontsize=8)
            plt.savefig(marginal_effect_exploration_folder + "/point_wise_marginal_effect_of_frequency_of_Medium_core_over_frequency_of_little_socket_number_of_thread_on_little_socket_and_8_th_core_frequency.png")
            plt.clf()
            plt.cla()
            plt.close()
            
        
    if phone_name == "google_pixel_4a_5g" :

        linear_coeff_vs_kernel_ridge_margins_file = marginal_effect_exploration_folder + "/linear_coeff_vs_kernel_ridge_margins.csv" # Can change depending on the r2 score
        if  X_format_in_model == "base_Y_N_on_socket":
            X_meaning_dictionnary = base_Y_N_on_socket__X_meaning_dictionnary
            utils.capture_kernel_means_marginal_and_linear_model_coeff(margins, linear_coefficients, linear_coeff_vs_kernel_ridge_margins_file, X_meaning_dictionnary)


            ### Plotting X_1 distribution plot (Note, it is the activation state of the first core! because we are in Base_Y format of X).
            # plotting histograph
        
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,0], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_little_socket_frequency_level.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,1], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_number_of_little_cores_actives.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,2], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_6_frequency_level.png", 3e-11, 3)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,3], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_7_frequency_level.png", 3e-11, 4)

            
            d_X_2_coefficients_file = marginal_effect_exploration_folder + "/d_X_2_linear_coefficients.csv"
            d_X_2_ols = sm.OLS(pointwise_margins[:,2], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_2_reg = d_X_2_ols.fit()
            d_X_2_linear_coefficients = d_X_2_reg.params
            print("X_2_d linear model parameters  = ",  d_X_2_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 2, 
                        d_X_i_linear_coefficients = d_X_2_linear_coefficients, 
                            file_path = d_X_2_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
        
            
            # plotting of d_X_2, regarding to other_variables with 
            _, (d_X_2_over_X_0, d_X_2_over_X_1, d_X_2_over_X_3) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 13))
            d_X_2_over_X_0.scatter(X_train[:,0], pointwise_margins[:,2], c = "blue")
            # Add title and axis names
            d_X_2_over_X_0.set_title('d_X_2 over X_0')
            d_X_2_over_X_0.set_xlabel('X_0 : frequency level of little socket')
            d_X_2_over_X_0.set_ylabel("d_X_2 : pointwise marginal effect of frequency of Medium core")
            d_X_2_over_X_0.tick_params(size=8)


            d_X_2_over_X_1.scatter(X_train[:,1], pointwise_margins[:,2],  c = "blue")
            # Add title and axis names
            d_X_2_over_X_1.set_title('d_X_2 over X_1')
            d_X_2_over_X_1.set_xlabel('X_1 : Number of threads on little socket')
            d_X_2_over_X_1.set_ylabel("d_X_2 ")
            d_X_2_over_X_1.tick_params(size=8)

        
            d_X_2_over_X_3.scatter(X_train[:,3], pointwise_margins[:,2],  c = "blue")
            # Add title and axis names
            d_X_2_over_X_3.set_title('d_X_2 over X_3')
            d_X_2_over_X_3.set_xlabel('X_3 : frequency of core 7 (8th core)')
            d_X_2_over_X_3.set_ylabel("d_X_2 : pointwise marginal effect of frequency of Medium core")
            d_X_2_over_X_3.tick_params(size=8)

            #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")

            plt.gcf().autofmt_xdate()
            plt.xticks(fontsize=8)
            plt.savefig(marginal_effect_exploration_folder + "/point_wise_marginal_effect_of_frequency_of_Medium_core_over_frequency_of_little_socket_number_of_thread_on_little_socket_and_8_th_core_frequency.png")
            plt.clf()
            plt.cla()
            plt.close()
            
        elif X_format_in_model == "base_Y":
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

            ### Plotting marginal effect plots
            """
            ## Regression of d_X_8 over all other variable including
            d_X_8_coefficients_file = marginal_effect_exploration_folder + "/d_X_8_linear_coefficients.csv"
            d_X_8_ols = sm.OLS(pointwise_margins[:,8], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_8_reg = d_X_8_ols.fit()
            d_X_8_linear_coefficients = d_X_8_reg.params
            print("X_8_d linear model parameters  = ",  d_X_8_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 8, 
                        d_X_i_linear_coefficients = d_X_8_linear_coefficients, 
                            file_path = d_X_8_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_8 over other variables")
            utils.plot_marginal_interactions(X_train, pointwise_margins, 8, 0, 1, 2, 3,4,5,6,7, 8, X_meaning_dictionnary, marginal_effect_exploration_folder)
        


            ## Regression of d_X_7 over all other variable including 
            d_X_7_coefficients_file = marginal_effect_exploration_folder + "/d_X_7_linear_coefficients.csv"
            d_X_7_ols = sm.OLS(pointwise_margins[:,7], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_7_reg = d_X_7_ols.fit()
            d_X_7_linear_coefficients = d_X_7_reg.params
            print("X_7_d linear model parameters  = ",  d_X_7_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 7, 
                        d_X_i_linear_coefficients = d_X_7_linear_coefficients, 
                            file_path = d_X_7_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_7 over other variables")
            utils.plot_marginal_interactions(X_train, pointwise_margins, 7, 0, 1, 2, 3,4,5,6,7, 8, X_meaning_dictionnary, marginal_effect_exploration_folder)
            
        

            ## Regression of d_X_0 over all other variable including 
            d_X_0_coefficients_file = marginal_effect_exploration_folder + "/d_X_0_linear_coefficients.csv"
            d_X_0_ols = sm.OLS(pointwise_margins[:,0], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_0_reg = d_X_0_ols.fit()
            d_X_0_linear_coefficients = d_X_0_reg.params
            print("X_0_d linear model parameters  = ",  d_X_0_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 0, 
                        d_X_i_linear_coefficients = d_X_0_linear_coefficients, 
                            file_path = d_X_0_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_0 over other variables")
            utils.plot_marginal_interactions(X_train, pointwise_margins, 0, 0, 1, 2, 3,4,5,6,7, 8, X_meaning_dictionnary, marginal_effect_exploration_folder)
            
            
            """
          
            

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
            
        elif X_format_in_model == "base_Y_F":
            X_meaning_dictionnary = base_Y_F__X_meaning_dictionnary
            utils.capture_kernel_means_marginal_and_linear_model_coeff(margins, linear_coefficients, linear_coeff_vs_kernel_ridge_margins_file, X_meaning_dictionnary)
            """
            {"X_0" : "Little Socket frequency is freed",
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
            """
            ### Plotting X_1 distribution plot (Note, it is the activation state of the first core! because we are in Base_Y format of X).
            # plotting histograph
        
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,0], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_little_socket_frequency_freed.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,1], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_little_socket_frequency_level.png", 3e-11, 8)

            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,2], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_0_state.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,9], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_6_frequency_level.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,11], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_core_7_frequency_level.png", 3e-11, 8)

            ### Plotting marginal effect plots

            ## Regression of d_X_0 (frequency of little socket is freed) over all other variable
            d_X_0_coefficients_file = marginal_effect_exploration_folder + "/d_X_0_linear_coefficients.csv"
            d_X_0_ols = sm.OLS(pointwise_margins[:,0], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_0_reg = d_X_0_ols.fit()
            d_X_0_linear_coefficients = d_X_0_reg.params
            print("X_0_d linear model parameters  = ",  d_X_0_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 0, 
                        d_X_i_linear_coefficients = d_X_0_linear_coefficients, 
                            file_path = d_X_0_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_0 over other variables")
            utils.plot_twelve_marginal_interactions(X_train, pointwise_margins, 0, 0, 1, 2, 3,4,5,6,7, 8, 9,10,11, X_meaning_dictionnary, marginal_effect_exploration_folder)
            
            


            ## Regression of d_X_8 over (frequency of medium socket is freed) all other variable including
            d_X_8_coefficients_file = marginal_effect_exploration_folder + "/d_X_8_linear_coefficients.csv"
            d_X_8_ols = sm.OLS(pointwise_margins[:,8], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_8_reg = d_X_8_ols.fit()
            d_X_8_linear_coefficients = d_X_8_reg.params
            print("X_8_d linear model parameters  = ",  d_X_8_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 8, 
                        d_X_i_linear_coefficients = d_X_8_linear_coefficients, 
                            file_path = d_X_8_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_8 over other variables")
            utils.plot_twelve_marginal_interactions(X_train, pointwise_margins, 8, 0, 1, 2, 3,4,5,6,7, 8, 9,10, 11, X_meaning_dictionnary, marginal_effect_exploration_folder)
        


            ## Regression of d_X_10 over (frequency of medium socket is freed) all other variable including
            d_X_10_coefficients_file = marginal_effect_exploration_folder + "/d_X_10_linear_coefficients.csv"
            d_X_10_ols = sm.OLS(pointwise_margins[:,10], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_10_reg = d_X_10_ols.fit()
            d_X_10_linear_coefficients = d_X_10_reg.params
            print("X_10_d linear model parameters  = ",  d_X_10_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 10, 
                        d_X_i_linear_coefficients = d_X_10_linear_coefficients, 
                            file_path = d_X_10_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_10 over other variables")
            utils.plot_twelve_marginal_interactions(X_train, pointwise_margins, 10, 0, 1, 2, 3,4,5,6,7, 8, 9,10,11, X_meaning_dictionnary, marginal_effect_exploration_folder)
                     
        elif X_format_in_model == "base_Y_F_N_on_socket":
            """
            base_Y_F_N_on_socket__X_meaning_dictionnary = {"X_0" : "Little Socket frequency is freed",
                                    "X_1" : "frequency level of Little Socket",
                                    "X_2" : "Number of little cores active",  
                                    "X_3" : "Medium Socket frequency is freed",
                                    "X_4" : "frequency level of Medium Socket or core 6",
                                    "X_5" : "Big Socket frequency is freed",
                                    "X_6" : "frequency level of Big Socket or core 7"} 
            """
            X_meaning_dictionnary = base_Y_F_N_on_socket__X_meaning_dictionnary
            utils.capture_kernel_means_marginal_and_linear_model_coeff(margins, linear_coefficients, linear_coeff_vs_kernel_ridge_margins_file, X_meaning_dictionnary)
           
            ### Plotting X_1 distribution plot (Note, it is the activation state of the first core! because we are in Base_Y format of X).
            # plotting histograph
        
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,0], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_little_socket_frequency_freed.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,1], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_little_socket_frequency_level.png", 3e-11, 8)

            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,2], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_number_of_little_cores_active.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,3], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_medium_socket_frequency_freed.png", 3e-11, 8)
            utils.plot_marginal_effect_histogramm_graph(pointwise_margins[:,5], marginal_effect_exploration_folder + "/point_wise_marginal_distribution_of_big_socket_frequency_freed.png", 3e-11, 8)

            ### Plotting marginal effect plots

            ## Regression of d_X_0 (frequency of little socket is freed) over all other variable
            d_X_0_coefficients_file = marginal_effect_exploration_folder + "/d_X_0_linear_coefficients.csv"
            d_X_0_ols = sm.OLS(pointwise_margins[:,0], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_0_reg = d_X_0_ols.fit()
            d_X_0_linear_coefficients = d_X_0_reg.params
            print("X_0_d linear model parameters  = ",  d_X_0_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 0, 
                        d_X_i_linear_coefficients = d_X_0_linear_coefficients, 
                            file_path = d_X_0_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_0 over other variables")
            utils.plot_seven_marginal_interactions(X_train, pointwise_margins, 0, 0, 1, 2, 3,4,5,6, X_meaning_dictionnary, marginal_effect_exploration_folder)
            
            


            ## Regression of d_X_8 over (frequency of medium socket is freed) all other variable including
            d_X_3_coefficients_file = marginal_effect_exploration_folder + "/d_X_3_linear_coefficients.csv"
            d_X_3_ols = sm.OLS(pointwise_margins[:,3], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_3_reg = d_X_3_ols.fit()
            d_X_3_linear_coefficients = d_X_3_reg.params
            print("X_3_d linear model parameters  = ",  d_X_3_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 3, 
                        d_X_i_linear_coefficients = d_X_3_linear_coefficients, 
                            file_path = d_X_3_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_3 over other variables")
            utils.plot_seven_marginal_interactions(X_train, pointwise_margins, 3, 0, 1, 2, 3,4,5,6, X_meaning_dictionnary, marginal_effect_exploration_folder)
        


            ## Regression of d_X_5 over (frequency of medium socket is freed) all other variable including
            d_X_5_coefficients_file = marginal_effect_exploration_folder + "/d_X_5_linear_coefficients.csv"
            d_X_5_ols = sm.OLS(pointwise_margins[:,5], X_train )  # warning in the sm OLS function argument format, y is the first parameter. 
            d_X_5_reg = d_X_5_ols.fit()
            d_X_5_linear_coefficients = d_X_5_reg.params
            print("X_5_d linear model parameters  = ",  d_X_5_linear_coefficients)
            utils.capture_d_X_i_linear_coefficient_on_others_X_variables(d_X_i_indice = 5, 
                        d_X_i_linear_coefficients = d_X_5_linear_coefficients, 
                            file_path = d_X_5_coefficients_file,
                        X_meaning_dictionnary_ = X_meaning_dictionnary)
            print("Plotting d_X_5 over other variables")
            utils.plot_seven_marginal_interactions(X_train, pointwise_margins, 5, 0, 1, 2, 3,4,5,6, X_meaning_dictionnary, marginal_effect_exploration_folder)
    
    
    return pointwise_margins, X_meaning_dictionnary      
  

       
       
       
        """
        # Caprices de Vlad
        print("--- Number of input with fourth core activated: " + repr(utils.count_number_of_input_with_fourth_core_on(X_train)))
        print("--- Size of X train: " + repr(len(X_train)))
        print("--- Ratio of input with fourth core activated: " + repr(float(utils.count_number_of_input_with_fourth_core_on(X_train)) / float(len(X_train))))
        
        input_to_look_at = utils.inputs_where_d_X_index_is_negative(pointwise_margins, 8, X_train)
        print("--- Inputs where d_X_8 is negative: " , repr(input_to_look_at))
        print("--- Size: ",  repr(len(input_to_look_at)) )
        print("--- Size of X train: " + repr(len(X_train)))
        """
        



    #print(" ********** d_X_1 np array:", pointwise_margins[:,0])
    #print(" ********** d_X_1 dataframe:", pandas.DataFrame(n_pointwise_margins[:,0]))
    #print(" ********** d_X_1 description: " + str (pandas.DataFrame(n_pointwise_margins[:,0]).describe()))

        """
        # plotting of d_X_1, regarding to other_variables with 
        _, (d_X_1_over_X_0, d_X_1_over_X_1, d_X_1_over_X_8) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 13))
        d_X_1_over_X_0.scatter(X_train[:,0], pointwise_margins[:,1], c = "blue")
        # Add title and axis names
        d_X_1_over_X_0.set_title('d_X_1 over X_0')
        d_X_1_over_X_0.set_xlabel('X_0 : frequency level of little socket')
        d_X_1_over_X_0.set_ylabel("d_X_1 : pointwise marginal effect of core 0 state")
        d_X_1_over_X_0.tick_params(size=8)


        d_X_1_over_X_1.scatter(X_train[:,1], pointwise_margins[:,1],  c = "blue")
        # Add title and axis names
        d_X_1_over_X_1.set_title('d_X_1 over X_1')
        d_X_1_over_X_1.set_xlabel('X_1 : state of core 0 (1rst core)')
        d_X_1_over_X_1.set_ylabel("d_X_1 ")
        d_X_1_over_X_1.tick_params(size=8)

    
        d_X_1_over_X_8.scatter(X_train[:,8], pointwise_margins[:,1],  c = "blue")
        # Add title and axis names
        d_X_1_over_X_8.set_title('d_X_1 over X_8')
        d_X_1_over_X_8.set_xlabel('X_8 : frequency of core 7 (8th core)')
        d_X_1_over_X_8.set_ylabel("d_X_1 : pointwise marginal effect of core 0 state")
        d_X_1_over_X_8.tick_params(size=8)

        #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")

        plt.gcf().autofmt_xdate()
        plt.xticks(fontsize=8)
        plt.savefig(marginal_effect_exploration_folder + "/point_wise_marginal_effect_of_core_1_state_over_little_socket__1_srt_and_8_th_cores.png")
        plt.clf()
        plt.cla()
        plt.close()

        print(" ********** d_X_1 np array:", n_pointwise_margins[:,0])
        print(" ********** d_X_1 dataframe:", pandas.DataFrame(n_pointwise_margins[:,0]))
        print(" ********** d_X_1 description: " + str (pandas.DataFrame(n_pointwise_margins[:,0]).describe()))

        """



        """
        # plotting of d_X_1, regarding to other_variables with 
        _, (d_X_1_over_X_0, d_X_1_over_X_1, d_X_1_over_X_8) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 13))
        d_X_1_over_X_0.scatter(X_train[:,0], pointwise_margins[:,1], c = "blue")
        # Add title and axis names
        d_X_1_over_X_0.set_title('d_X_1 over X_0')
        d_X_1_over_X_0.set_xlabel('X_0 : frequency level of little socket')
        d_X_1_over_X_0.set_ylabel("d_X_1 : pointwise marginal effect of core 0 state")
        d_X_1_over_X_0.tick_params(size=8)


        d_X_1_over_X_1.scatter(X_train[:,1], pointwise_margins[:,1],  c = "blue")
        # Add title and axis names
        d_X_1_over_X_1.set_title('d_X_1 over X_1')
        d_X_1_over_X_1.set_xlabel('X_1 : state of core 0 (1rst core)')
        d_X_1_over_X_1.set_ylabel("d_X_1 ")
        d_X_1_over_X_1.tick_params(size=8)

    
        d_X_1_over_X_8.scatter(X_train[:,8], pointwise_margins[:,1],  c = "blue")
        # Add title and axis names
        d_X_1_over_X_8.set_title('d_X_1 over X_8')
        d_X_1_over_X_8.set_xlabel('X_8 : frequency of core 7 (8th core)')
        d_X_1_over_X_8.set_ylabel("d_X_1 : pointwise marginal effect of core 0 state")
        d_X_1_over_X_8.tick_params(size=8)

        #_ = d_X_0_over_X_5.set_title("Point wise marginal effect of frequency of core 0 according to the one of core 1, 2 and 3")

        plt.gcf().autofmt_xdate()
        plt.xticks(fontsize=8)
        plt.savefig(marginal_effect_exploration_folder + "/point_wise_marginal_effect_of_core_1_state_over_little_socket__1_srt_and_8_th_cores.png")
        plt.clf()
        plt.cla()
        plt.close()

        print(" ********** d_X_1 np array:", n_pointwise_margins[:,0])
        print(" ********** d_X_1 dataframe:", pandas.DataFrame(n_pointwise_margins[:,0]))
        print(" ********** d_X_1 description: " + str (pandas.DataFrame(n_pointwise_margins[:,0]).describe()))

        """








"""
####################################  Prediction on samsung galaxy s8
X_user_friendly = data.X_user_friendly_samsung_galaxy_s8
print ("*** Total configurations user friendly: ", X_user_friendly)
X =data.X_samsung_galaxy_s8
print ("*** Total Configurations formatted: ", X)
X_dict = data.X_dict_samsung_galaxy_s8
print ("*** Total Configurations dictionnary: ", X_dict)
y = data.y_samsung_galaxy_s8
print("*** Ratio energy by wokload : ", y)




# to do generate_equivalent_entries(X,y)
#################################


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=2)
X_user_friendly_train = utils.get_X_user_friendly_from_X(X_train, X_dict)
X_user_friendly_test = utils.get_X_user_friendly_from_X(X_test, X_dict)


print ("Train set Configurations : ", X_train)
print ("Train set energy by workload : ", y_train)
print ("Test set Configurations : ", X_test)
print ("Test set energy by workload : ", y_test)
print ("Train set Configurations in user friendly mode : ", X_user_friendly_train)
print ("Test set Configurations in user friendly mode : ", X_user_friendly_test)


############## now using kernel ridge to train data
krr = KernelRidge(alpha=1.0, kernel="rbf") #gamma=10)
krr.fit(X_train, y_train)
krr_y_test = krr.predict(X_test)


_, (orig_data_ax, testin_data_ax, kernel_ridge_ax) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 13))
orig_data_ax.bar(X_user_friendly_train,y_train, width=0.4)
# Add title and axis names
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

kernel_ridge_ax.bar(X_user_friendly_test,krr_y_test, width=0.4)
# Add title and axis names
kernel_ridge_ax.set_title('Predited energy/workload ratio')
kernel_ridge_ax.set_xlabel('Configuration')
kernel_ridge_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
kernel_ridge_ax.tick_params(size=8)

_ = kernel_ridge_ax.set_title("Predicted data\n using kernel ridge, R2 = " + str(krr.score(X_test, y_test)))

print("error = ", krr.score(X_test, y_test))
print("parrams = " ,  krr.get_params(False))
print("regressors = " ,  krr)
plt.gcf().autofmt_xdate()
plt.xticks(fontsize=8)
plt.savefig("kernel_ridge_prediction_on_samsung_galaxy_s8.png")
plt.clf()
plt.cla()
plt.close()
"""

####### This code was used to test the avalaible source code of the statmodels class  kernridgeregress_class from this
# https://github.com/statsmodels/statsmodels/blob/825581cf17f1e79118592f15f49be7ad890a7104/statsmodels/sandbox/regression/kernridgeregress_class.py#L9

"""
 # code coming from sklearn
krr = KernelRidge(alpha=1.0, kernel="rbf") #gamma=10)
krr.fit(X_train, y_train)
krr_y_test = krr.predict(X_test)




#Code coming from the stasmodels kernelrige class source code
m,k = 10,4
##m,k = 50,4
upper = 6
scale = 10
xs = np.linspace(1,upper,m)[:,np.newaxis]
#xs1 = xs1a*np.ones((1,4)) + 1/(1.0+np.exp(np.random.randn(m,k)))
#xs1 /= np.std(xs1[::k,:],0)   # normalize scale, could use cov to normalize
##y1true = np.sum(np.sin(xs1)+np.sqrt(xs1),1)[:,np.newaxis]
xs1 = np.sin(xs)#[:,np.newaxis]
y1true = np.sum(xs1 + 0.01*np.sqrt(np.abs(xs1)),1)[:,np.newaxis]
y1 = y1true + 0.10 * np.random.randn(m,1)

stride = 3 #use only some points as trainig points e.g 2 means every 2nd
xstrain = xs1[::stride,:]
ystrain = y1[::stride,:]
ratio = int(m/2)
print("ratio = ", ratio)
xstrain = np.r_[xs1[:ratio,:], xs1[ratio+10:,:]]
ystrain = np.r_[y1[:ratio,:], y1[ratio+10:,:]]
index = np.hstack((np.arange(m/2), np.arange(m/2+10,m)))
print("Their own X", xstrain)

# added for standartization
xstrain_ = utils.standartize(xstrain)
print("Their own X", xstrain)
print ("Standartized X", xstrain_)
xstrain = xstrain_
#end of standartization

# added for lambda exploration
utils.find_regularization_parameter(xstrain, ystrain)
# end added for lambda exploration 




print("Their own y", ystrain)
gp1 = GaussProcess(xstrain, ystrain, #kernel=kernel_euclid,
                   ridgecoeff=5*1e-4)
gp1.fit(ystrain)
krr_y_test = gp1.predict(np.asarray(xstrain))
print("Predicted y test = ", krr_y_test)
print(" Computed c values  = ", gp1.parest)
c_vector = gp1.parest
sigma_2 = 0.5
X = xstrain
#End of code coming from the stasmodels kernelrige class source code






 
 n_samples, n_features = 10, 2
rng = np.random.RandomState(0)
#y = rng.randn(n_samples)
y = np.random.randint(1,9,n_samples)
X = rng.randn(n_samples, n_features)
print("X", X)
X_error =  X[:,np.newaxis,:]
print("X_error", X_error)

gauss_process = GaussProcess(X, y,
                   ridgecoeff=5*1e-4)

krr = gauss_process.fit(y) 

"""

####### this code was used to debug the data set splitting 
"""
_, (train_ax, test_ax) = plt.subplots(nrows=2,  sharex=True, sharey=True, figsize=(11, 10))


train_ax.bar(X_user_friendly_train,y_train, width=0.4)
# Add title and axis names
train_ax.set_title('Energy/ Workload according to the configuration')
train_ax.set_xlabel('Number of CPUs')
train_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
train_ax.tick_params( size=8)

test_ax.bar(X_user_friendly_test,y_test, width=0.4)
# Add title and axis names
test_ax.set_title('Energy/ Workload according to the configuration')
test_ax.set_xlabel('Number of CPUs')
test_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
test_ax.tick_params(size=8)

_ = test_ax.set_title("Testing data")


plt.gcf().autofmt_xdate()
plt.xticks(fontsize=8)

plt.savefig("kernel_ridge_training_and_testing_configuration_data__plot.png")

plt.clf()
plt.cla()
plt.close()
"""

########### adated version - before integrating powertool results
"""
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import numpy as np
n_samples, n_features = 100, 2
rng = np.random.RandomState(0)
#y = rng.randn(n_samples)
y = np.random.randint(1,9,n_samples)

X = rng.randn(n_samples, n_features)

print("X values")
print (X)

print("y values")
print(y)





# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=2)

_, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))


train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
train_ax.set_ylabel("Y values")
train_ax.set_xlabel("X values")
train_ax.set_title("Training data")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel("X values")
_ = test_ax.set_title("Testing data")



plt.savefig("kernel_ridge_training_and_testing_random_data__plot.png")

plt.clf()
plt.cla()
plt.close()

############## now using kernel ridge to train data

krr = KernelRidge(alpha=1.0) #gamma=10)
krr.fit(X_train, y_train)

krr_y_test = krr.predict(X_test)


# %%
fig, (orig_data_ax, testin_data_ax, kernel_ridge_ax) = plt.subplots(
    ncols=3, figsize=(14, 4)
)


orig_data_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
orig_data_ax.set_ylabel("Y values")
orig_data_ax.set_xlabel("X values")
orig_data_ax.set_title("training data")

testin_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
testin_data_ax.set_ylabel("Y tested values")
testin_data_ax.set_xlabel("X tested values")
testin_data_ax.set_title("testing datas")

kernel_ridge_ax.scatter(X_test[:, 0], X_test[:, 1], c=krr_y_test)
kernel_ridge_ax.set_ylabel("Y predicted values on tested sample")
kernel_ridge_ax.set_xlabel("X tested values")
_ = kernel_ridge_ax.set_title("Projection of testing data\n using kernel ridge")

print("error = ", krr.score(X_test, krr_y_test))

plt.savefig("kernel_ridge_training_testing_predict_on_random_data__plot.png")

plt.clf()
plt.cla()
plt.close()
"""



############ unmodified version 
"""
from sklearn.kernel_ridge import KernelRidge
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
krr = KernelRidge(alpha=1.0)
krr.fit(X, y)
param = krr.get_params(1)
print(param)
"""
