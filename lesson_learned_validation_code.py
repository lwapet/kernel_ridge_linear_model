# this file is for the validation of the kernel ridge model,
# In this file we saved all sqlite queries used to valide lesson learned
# each function in this file is associated to one lesson learned and contains sql queries for the lesson learned validation. 
# each function return a string to add to the lesson_learned_validation_summary excel file 
# the lesson_learned_validation_summary excel file formatted like this:
# variable to increase, chipset state, suitable-contraindicated-or-neutral,  validation score, [accepted transitions], [rejected transitions]   

# the validation error variable is defined according to visual aspect of the graph
# need to compute the exact error propagation in the future if possible. 
validation_error = 0 # 1e+9  # 1E9



def compute_static_score(conn, acceptability_degree = 10):
    # Let suppose that acceptability_degree = N
    # Validation of the ability of the kernel ridge to capture best configurations. 
    # Intersection between first N configurations in measurement and first N  of the same configurations when estimated by the model.   
    
    number_of_measurement = 0
    command = '''
    SELECT COUNT(*) FROM
       (SELECT *
        FROM  configuration_measurements
        AS test_set)
    '''
    cursor = conn.execute(command)
    print ("--- number of measurements:")
    for row in cursor:
        print("value = ", row[0])
        number_of_measurement = row[0]
    if acceptability_degree > number_of_measurement:
        acceptability_degree = number_of_measurement
    
    # to print the first configurations, can be usefull, if we whant to mention it in the paper
    command = ''' 
    SELECT * FROM
       (SELECT * FROM configuration_measurements
        INNER JOIN 
            ((SELECT configuration_efficiency_estimation.configuration_id
             FROM  configuration_efficiency_estimation WHERE configuration_efficiency_estimation.train_or_test_set == "test")
             AS test_set)
        USING(configuration_id)  
        ORDER BY energy_efficiency DESC LIMIT ''' + str(acceptability_degree) + ''') AS configuration_mesured_ordered
    INNER JOIN
        (SELECT *
        FROM  configuration_efficiency_estimation WHERE configuration_efficiency_estimation.train_or_test_set == "test"
        ORDER BY  energy_efficiency  DESC LIMIT ''' + str(acceptability_degree) + ''') AS configuration_estimated_ordered
    USING(configuration_id)
    INNER JOIN configuration_representations
    USING(configuration_id)
    '''
   
    #command = '''SELECT * FROM configuration_description__google_pixel_4a_5g INNER JOIN configuration_measurements USING(configuration_id);''' 
    cursor = conn.execute(command)
    print ("--- intersection result for test set:")
    i = 0
    for row in cursor:
        i = i + 1
        print("value = ", row)
        if i > 5:
            break


    # to compute the precision
    command = '''  
    SELECT COUNT(*) FROM
       (SELECT * FROM configuration_measurements
        INNER JOIN 
            ((SELECT configuration_efficiency_estimation.configuration_id
             FROM  configuration_efficiency_estimation WHERE configuration_efficiency_estimation.train_or_test_set == "test")
             AS test_set)
        USING(configuration_id)  
        ORDER BY energy_efficiency DESC LIMIT ''' + str(acceptability_degree) + ''') AS configuration_mesured_ordered
    INNER JOIN
        (SELECT *
        FROM  configuration_efficiency_estimation WHERE configuration_efficiency_estimation.train_or_test_set == "test"
        ORDER BY  energy_efficiency  DESC LIMIT ''' + str(acceptability_degree) + ''') AS configuration_estimated_ordered
    USING(configuration_id)
    '''
   
    #command = '''SELECT * FROM configuration_description__google_pixel_4a_5g INNER JOIN configuration_measurements USING(configuration_id);''' 
    cursor = conn.execute(command)
    print ("--- intersection result for test set:")
    
    for row in cursor:
        print("value = ", row[0])
        print("precision = ", row[0]/acceptability_degree)
        precision =  row[0]/acceptability_degree
    return   precision

    """ Without considering test set
    command = '''
    SELECT COUNT(*) FROM
        ((SELECT * FROM configuration_measurements ORDER BY  energy_efficiency  DESC LIMIT ''' + str(acceptability_degree) + ''') AS configuration_measurements_ordered
        INNER JOIN 
        (SELECT * FROM
            (SELECT * FROM  configuration_efficiency_estimation INNER JOIN configuration_measurements USING(configuration_id))
        ORDER BY  energy_efficiency  DESC LIMIT ''' + str(acceptability_degree) + ''') AS configuration_estimation_of_the_one_mesured__ordered
        USING(configuration_id))
    INNER JOIN
        configuration_representations
    USING(configuration_id)
    '''
    cursor = conn.execute(command)
    print ("--- intersection result for all the experiment set:")
    for row in cursor:
        print("value = ", row[0])
        print("precision = ", row[0]/acceptability_degree)
    """

    """ to list configurations
    command = '''  
        SELECT * FROM configuration_measurements
        INNER JOIN 
            ((SELECT configuration_efficiency_estimation.configuration_id
             FROM  configuration_efficiency_estimation WHERE configuration_efficiency_estimation.train_or_test_set == "test")
             AS test_set)
        USING(configuration_id)  
        ORDER BY energy_efficiency DESC LIMIT ''' + str(acceptability_degree) 
    
    print("----- Measured set ordered")
    cursor = conn.execute(command)
    for row in cursor:
        print("row = ", row)
    command = '''  
        SELECT *
        FROM  configuration_efficiency_estimation WHERE configuration_efficiency_estimation.train_or_test_set == "test"
        ORDER BY  energy_efficiency  DESC LIMIT ''' + str(acceptability_degree) 
  
    
    print("----- estimated  set ordered")
    #command = '''SELECT * FROM configuration_description__google_pixel_4a_5g INNER JOIN configuration_measurements USING(configuration_id);''' 
    cursor = conn.execute(command)
    for row in cursor:
        print("row = ", row)
    """ 
    
        





def compute_decision_score(expected_efficiency_behavior, variation, transition_as_string, score_variation,  score, accepted_transitions, rejected_transitions):
    # score_variation is the delta value to add to the score,
    # it depends on the nomber of subpart n of the lessone learned. 
    # it is normally 100/n
    # the last three parameters are the previous values that we modify in this function
    if(expected_efficiency_behavior == "decreases"):
        if(variation < 0):
            score = score + score_variation
            accepted_transitions = accepted_transitions +  transition_as_string + "["+ repr(variation*1E-9)+  "]; "
        elif(variation > 0):
            rejected_transitions = rejected_transitions + transition_as_string + "["+ repr(variation*1E-9)+  "]; "
    elif(expected_efficiency_behavior == "increases"):
        if(variation < 0):
            rejected_transitions = rejected_transitions + transition_as_string + "["+ repr(variation*1E-9)+  "]; "
        elif(variation > 0):
            score = score + score_variation
            accepted_transitions = accepted_transitions + transition_as_string + "["+ repr(variation*1E-9)+  "]; "
    elif(expected_efficiency_behavior == "is stable"):
        if(abs(variation) < validation_error):
            score = score + score_variation
            accepted_transitions = accepted_transitions +  transition_as_string + "["+ repr(variation*1E-9)+  "]; "
        elif(variation > 0):
            rejected_transitions = rejected_transitions + transition_as_string + "["+ repr(variation*1E-9)+  "]; "
    return score, accepted_transitions, rejected_transitions



def  generate_increments_couples(possible_values):
    #this function takes as input an array and generate couple of transition, from the first element of the array to second ones
    # if input is 0,1,2, it will generate 0,1 and 1,2  
    result = []
    print("--- Computing increment couple on possible values", possible_values)
    if len(possible_values) < 2: 
        print ("Error! cannot find transitions couples of arrays containing one element")
    for i in range(0,len(possible_values) -1 ):
        result.append([possible_values[i], possible_values[i+1] ])
        print("--- Computing increment couple on possible values; result ", result)
    return result



def validate_fitsAll_interaction_table(conn, cibled_column, possible_values, secondary_columns,  avg_marginal_score_table): # the possible_values foreeach column is alreay in the computed marginal score table
    # compute dynamic score of each entry in the table
    result_table = []
    
    for secondary_column, entry_in_table in zip(secondary_columns, avg_marginal_score_table): #   l column name,     [possible values of L] -> [marginal AVG scores of those values]
        possibles_L_values = entry_in_table [0]
        avg_marginal_scores = entry_in_table [1]
        fitsAll_dynamic_scores_summary = []
        lth_index = 0
        for L_value,avg_marginal_score in zip(possibles_L_values, avg_marginal_scores): 
            fitsAll_dynamic_scores_summary.append(validate_fitsAll_advice(conn, cibled_column, possible_values, secondary_column, str(L_value), avg_marginal_score))


        result_table.append([possibles_L_values,avg_marginal_scores,fitsAll_dynamic_scores_summary])

    return result_table      


def validate_fitsAll_advice(conn, cibled_column, possible_values, secondary_column, L_value, avg_marginal_score):
    print("--- Evaluating FitsAll advice for cibled column " + cibled_column + " and secondary column " + secondary_column + " having value ", L_value)
    
    J_increment_couples = generate_increments_couples(possible_values)
    positive_differences = []
    negative_differences = []
    for couple in J_increment_couples:
        J_initial_value = couple[0]
        J_final_value = couple[1]
        difference = validate_fitsAll_increment (conn, cibled_column, J_initial_value, J_final_value, secondary_column, L_value)
        if difference is not None:      
            if difference < 0:
                negative_differences.append([J_initial_value, J_final_value, difference])
            else: 
                positive_differences.append([J_initial_value, J_final_value, difference])
               
    
    number_of_computed_differences = (len(negative_differences) + len(positive_differences)) # we only consider computed differences, we don't count cases where the increment was impossible. 
    fitsAll_dynamic_score = None
    if number_of_computed_differences > 0:
        if avg_marginal_score < 0:
            fitsAll_dynamic_score = len(negative_differences)/number_of_computed_differences * 100
        else: 
            fitsAll_dynamic_score = len(positive_differences)/number_of_computed_differences * 100
    else: 
        fitsAll_dynamic_score = None
    print("--- Positive differences ", positive_differences)
    print("--- Negative differences ", negative_differences)
    print("--- Fitstall dynamic score on this advice ", fitsAll_dynamic_score)
    return [fitsAll_dynamic_score, negative_differences, positive_differences]


def validate_fitsAll_increment(conn, cibled_column, J_initial_value, J_final_value, secondary_column, L_value):
    # This function help to validate a transition adviced by fitsall,
    # see the paper for description, the cibled column we whant to increment is J, from value initial_value to value final_value. 
    # the secondary column is the colum L which can be a suitable case for the interaction J-L
    # Note: values should be passed as strings!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # and if they are suppose to be string in the sql request add the symbol "" in the string
    print("--- Evaluating FitsAll increment for cibled column " + cibled_column + " going from " +J_initial_value+ " to " +J_final_value+ " \n and secondary column " + secondary_column + " having value ", L_value)
    initial_value = J_initial_value 
    final_value = J_final_value
    command = '''
    SELECT
        avg(configuration_description_measurements_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + initial_value + '''.energy_efficiency),
        avg(configuration_description_measurements_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + final_value + '''.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + initial_value + '''.configuration_id,
            configuration_description_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + initial_value + '''.''' + cibled_column + ''',
            configuration_description_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + initial_value + '''.''' + secondary_column + '''
        FROM
                (SELECT
                    configuration_description_''' + secondary_column + '''__'''+ L_value + '''.configuration_id,
                    configuration_description_''' + secondary_column + '''__'''+ L_value + '''.''' + cibled_column + ''',
                    configuration_description_''' + secondary_column + '''__'''+ L_value + '''.''' + secondary_column + '''
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.''' + cibled_column + ''',
                        configuration_description__google_pixel_4a_5g.''' + secondary_column + '''
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.''' + secondary_column + ''' == ''' + L_value +''' ) /* see if we can reduce this */  
                    AS configuration_description_''' + secondary_column + '''__'''+ L_value + '''
                WHERE
                    configuration_description_''' + secondary_column + '''__'''+ L_value + '''.''' + cibled_column+ ''' == ''' + initial_value + ''') 
                AS  configuration_description_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + initial_value + '''
        INNER JOIN 
                configuration_measurements       
        USING(configuration_id)  
        
        INNER JOIN 
            ((SELECT configuration_efficiency_estimation.configuration_id
             FROM  configuration_efficiency_estimation WHERE configuration_efficiency_estimation.train_or_test_set == "test")
             AS test_set)
        USING(configuration_id))  

        AS configuration_description_measurements_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + initial_value + ''', 
        


        /*end values*/     
         (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + final_value + '''.configuration_id,
            configuration_description_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + final_value + '''.''' + cibled_column + ''',
            configuration_description_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + final_value + '''.''' + secondary_column + '''
        FROM
                (SELECT
                    configuration_description_''' + secondary_column + '''__'''+ L_value + '''.configuration_id,
                    configuration_description_''' + secondary_column + '''__'''+ L_value + '''.''' + cibled_column + ''',
                    configuration_description_''' + secondary_column + '''__'''+ L_value + '''.''' + secondary_column + '''
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.''' + cibled_column + ''',
                        configuration_description__google_pixel_4a_5g.''' + secondary_column + '''
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.''' + secondary_column + ''' == ''' + L_value +''' ) /* see if we can reduce this */  
                    AS configuration_description_''' + secondary_column + '''__'''+ L_value + '''
                WHERE
                    configuration_description_''' + secondary_column + '''__'''+ L_value + '''.''' + cibled_column+ ''' == ''' + final_value + ''') 
                AS  configuration_description_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + final_value + '''
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id) 

        INNER JOIN 
            ((SELECT configuration_efficiency_estimation.configuration_id
             FROM  configuration_efficiency_estimation WHERE configuration_efficiency_estimation.train_or_test_set == "test")
             AS test_set)
        USING(configuration_id))  

        AS configuration_description_measurements_''' + secondary_column + '''_is_'''+ L_value + '''_and_''' + cibled_column+ '''_is_''' + final_value + '''; '''



    cursor = conn.execute(command)
    difference = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        if  row[0] is None or row[1] is None:
            return None
        print("difference ", row[1] - row[0])
        difference =  row[1] - row[0]

    return difference









def validate__scheduling_thread_on_core_i_when_condition_on_socket_frequency(i, socket_type, frequency_levels,conn,expected_efficiency_behavior ):
    # test if we schedule thread on core "i"
    # if the socket "socket_type" has the frequency "frequency_levels", frequency levels should be  delimited with the key word "or" !!!! 
    variable_to_increase = "core " + str(i) + " state" 
    suitable_contraindication_or_neutral = "suitable - efficiency should increase" if expected_efficiency_behavior == "increases" else "contraindicated -  efficiency should decrease" if expected_efficiency_behavior == "decreases" else "neutral - efficiency should be stable" 
    chipset_state = socket_type + " socket frequency is " + frequency_levels 
    
    validation_score = "NULL" #(Validated - or Not Validated  [Score%])
    score = 0 # from 0 to 100
    accepted_transitions = ""
    rejected_transitions = "" 
    
    number_of_cases = 0
    all_frequency_levels = [level.strip() for level in frequency_levels.split("or")] 
    score_variation = 100 / len(all_frequency_levels)  # to modify

    if socket_type == "little":
        socket_state_table_column_name = "little_socket_frequency" 
        level_to_int_dictionnary = {"low": 0, "medium": 1, "high": 2}
    elif socket_type == "medium":
        socket_state_table_column_name = "core_6_state_freq_level"
        level_to_int_dictionnary = {"low": 1, "medium": 2, "high": 3}
    elif socket_type == "big":
        socket_state_table_column_name = "core_7_state_freq_level"
        level_to_int_dictionnary = {"low": 1, "medium": 2, "high": 3}


    for level in all_frequency_levels:
        number_of_cases = number_of_cases + 1
        lesson_learned_description = ''' 
            Lesson learne: scheduling thread on core ''' + str(i) + ''' 
            part '''+ str(number_of_cases) + ''' :  '''  + socket_type + ''' frequency level is ''' + level + '''
            energy efficiency should increase
        '''
        print("--- Lesson learned:" , lesson_learned_description)
    
        command = '''
        SELECT
            avg(configuration_description_measurements_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_OFF.energy_efficiency),
            avg(configuration_description_measurements_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_ON.energy_efficiency)
        FROM
            /*initial values*/     
            (SELECT
                configuration_measurements.energy_efficiency,
                configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_OFF.configuration_id,
                configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state,
                configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_OFF.''' + socket_state_table_column_name + '''
            FROM
                    (SELECT
                        configuration_description_''' + socket_type + '''_socket_freq_'''+ level+ '''.configuration_id,
                        configuration_description_''' + socket_type + '''_socket_freq_'''+ level+ '''.core_''' + str(i) + '''_state,
                        configuration_description_''' + socket_type + '''_socket_freq_'''+ level+ '''.''' + socket_state_table_column_name + '''
                    FROM
                        (SELECT
                            configuration_description__google_pixel_4a_5g.configuration_id,
                            configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state,
                            configuration_description__google_pixel_4a_5g.''' + socket_state_table_column_name + '''
                        FROM
                            configuration_description__google_pixel_4a_5g
                        WHERE
                            configuration_description__google_pixel_4a_5g.''' + socket_state_table_column_name + '''== ''' + str(level_to_int_dictionnary[level]) +''' ) /* see if we can reduce this */  
                        AS configuration_description_''' + socket_type + '''_socket_freq_'''+ level+ '''
                    WHERE
                        configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''.core_''' + str(i) + '''_state == 0) 
                    AS  configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_OFF
            INNER JOIN 
                    configuration_measurements 
            USING(configuration_id))  
            AS configuration_description_measurements_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_OFF, 
            


            /*end values*/     
            (SELECT
                configuration_measurements.energy_efficiency,
                configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_ON.configuration_id,
                configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state,
                configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_ON.''' + socket_state_table_column_name + '''
            FROM
                    (SELECT
                        configuration_description_''' + socket_type + '''_socket_freq_'''+ level+ '''.configuration_id,
                        configuration_description_''' + socket_type + '''_socket_freq_'''+ level+ '''.core_''' + str(i) + '''_state,
                        configuration_description_''' + socket_type + '''_socket_freq_'''+ level+ '''.''' + socket_state_table_column_name + '''
                    FROM
                        (SELECT
                            configuration_description__google_pixel_4a_5g.configuration_id,
                            configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state,
                            configuration_description__google_pixel_4a_5g.''' + socket_state_table_column_name + '''
                        FROM
                            configuration_description__google_pixel_4a_5g
                        WHERE
                            configuration_description__google_pixel_4a_5g.''' + socket_state_table_column_name + '''== ''' + str(level_to_int_dictionnary[level]) +''' ) /* see if we can reduce this */  
                        AS configuration_description_''' + socket_type + '''_socket_freq_'''+ level+ '''
                    WHERE
                        configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''.core_''' + str(i) + '''_state == 1) 
                    AS  configuration_description_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_ON
            INNER JOIN 
                    configuration_measurements 
            USING(configuration_id))  
            AS configuration_description_measurements_''' + socket_type + '''_socket_freq_''' + level + '''_core_''' + str(i) + '''_ON; '''



        cursor = conn.execute(command)
        print (lesson_learned_description)
        variation = 0
        for row in cursor:
            print("before = ", row[0])
            print("after = ", row[1])
            print("variation ", row[1] - row[0])
            variation =  row[1] - row[0]

        score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "0 -> 1" , score_variation,
                                                                             score, accepted_transitions, rejected_transitions)
    
    
    validation_score = str(score) + " %" 
    # variable to increase, chipset state,  suitable-contraindicated-or-neutral, validation score, [accepted transition(s)], [rejected transition(s)]   
    return variable_to_increase + "," + chipset_state + "," + suitable_contraindication_or_neutral + ","  + \
                    validation_score + "," + accepted_transitions + "," + rejected_transitions

    # next to test like this







def validate__scheduling_thread_on_medium_or_big_core_i_no_matter_little_core_j_state( i, j, conn, expected_efficiency_behavior ): # conn is the connection handle to the data base
    variable_to_increase = "core " + str(i) + " state" 
    suitable_contraindication_or_neutral = "suitable - efficiency should increase" if expected_efficiency_behavior == "increases" \
        else "contraindicated -  efficiency should decrease" if expected_efficiency_behavior == "decreases" \
        else "neutral - efficiency should be stable" 
    chipset_state = "Core " + str(j) + " is ON or OFF"
    
    validation_score = "NULL" #(Validated - or Not Validated  [Score%])
    score = 0 # from 0 to 100
    score_variation = 100/6
    accepted_transitions = ""
    rejected_transitions = ""     

    lesson_learned_description = ''' 
        Lesson learne: scheduling thread on core ''' + str(i) + ''' 
        part 1 : core_''' + str(j) + '''_state = 0
        energy efficiency should increase
    '''
   
    command = '''
    SELECT
        avg(configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.energy_efficiency),
        avg(configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_Low.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.configuration_id,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_OFF.configuration_id,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_OFF
                WHERE
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level == 0) 
                AS  configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.configuration_id,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_OFF.configuration_id,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_OFF
                WHERE
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level == 1) 
                AS  configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_Low; '''



    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "0 -> 1" , score_variation,
                                                                             score, accepted_transitions, rejected_transitions)
    



    lesson_learned_description = ''' 
        Lesson learne: core ''' + str(i) + ''' frequency transition is 1->2
        part 2 : core_''' + str(j) + '''_state = 0
        energy efficiency should increase
    '''

 

    command = '''
    SELECT
        avg(configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_Low.energy_efficiency),
        avg(configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_Medium.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.configuration_id,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_OFF.configuration_id,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_OFF
                WHERE
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level == 1) 
                AS  configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_Low, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.configuration_id,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_OFF.configuration_id,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_OFF
                WHERE
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level == 2) 
                AS  configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_Medium; '''






    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "1 -> 2" , score_variation,
                                                                             score, accepted_transitions, rejected_transitions)
    


    
    lesson_learned_description = ''' 
        Lesson learne: core ''' + str(i) + ''' frequency transition is 2->3
        part 3 : core_''' + str(j) + '''_state = 0
        energy efficiency should increase
    '''




    command = '''
    SELECT
        avg(configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_Medium.energy_efficiency),
        avg(configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_High.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.configuration_id,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_OFF.configuration_id,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_OFF
                WHERE
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level == 2) 
                AS  configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_Medium, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.configuration_id,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_OFF.configuration_id,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_OFF
                WHERE
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state_freq_level == 3) 
                AS  configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_freq_High; '''






    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "2 -> 3" , score_variation,
                                                                             score, accepted_transitions, rejected_transitions)
    







    lesson_learned_description = ''' 
        Lesson learne: scheduling thread on core ''' + str(i) + ''' 
        part 4 : core_''' + str(j) + '''_state = 1
        energy efficiency should increase
    '''
   
    command = '''
    SELECT
        avg(configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.energy_efficiency),
        avg(configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_Low.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.configuration_id,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_ON.configuration_id,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_ON
                WHERE
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level == 0) 
                AS  configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.configuration_id,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_ON.configuration_id,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_ON
                WHERE
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level == 1) 
                AS  configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_Low; '''



    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "0 -> 1" , score_variation,
                                                                             score, accepted_transitions, rejected_transitions)
    



    lesson_learned_description = ''' 
        Lesson learne: core ''' + str(i) + ''' frequency transition is 1->2
        part 5 : core_''' + str(j) + '''_state = 1
        energy efficiency should increase
    '''

 

    command = '''
    SELECT
        avg(configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_Low.energy_efficiency),
        avg(configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_Medium.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.configuration_id,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_ON.configuration_id,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_ON
                WHERE
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level == 1) 
                AS  configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_Low, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.configuration_id,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_ON.configuration_id,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_ON
                WHERE
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level == 2) 
                AS  configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_Medium; '''






    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "1 -> 2" , score_variation,
                                                                             score, accepted_transitions, rejected_transitions)
    


    
    lesson_learned_description = ''' 
        Lesson learne: core ''' + str(i) + ''' frequency transition is 2->3
        part 6 : core_''' + str(j) + '''_state = 1
        energy efficiency should increase
    '''




    command = '''
    SELECT
        avg(configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_Medium.energy_efficiency),
        avg(configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_High.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.configuration_id,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_ON.configuration_id,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_ON
                WHERE
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level == 2) 
                AS  configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_Medium, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.configuration_id,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state_freq_level,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_ON.configuration_id,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state_freq_level,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_ON
                WHERE
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state_freq_level == 3) 
                AS  configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_freq_High; '''






    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "2 -> 3" , score_variation,
                                                                             score, accepted_transitions, rejected_transitions)
    



    

    validation_score = str(score) + " %" 
    # variable to increase, chipset state,  suitable-contraindicated-or-neutral, validation score, [accepted transition(s)], [rejected transition(s)]   
    return variable_to_increase + "," + chipset_state + "," + suitable_contraindication_or_neutral + ","  + \
                    validation_score + "," + accepted_transitions + "," + rejected_transitions























def validate__scheduling_thread_on_little_core_i_no_matter_core_j_state( i, j, conn, expected_efficiency_behavior ): # conn is the connection handle to the data base
    variable_to_increase = "core " + str(i) + " state" 
    suitable_contraindication_or_neutral = "suitable - efficiency should increase" if expected_efficiency_behavior == "increases" else "contraindicated -  efficiency should decrease" if expected_efficiency_behavior == "decreases" else "neutral - efficiency should be stable" 
    chipset_state = "Core " + str(j) + " is ON or OFF"
    
    validation_score = "NULL" #(Validated - or Not Validated  [Score%])
    score = 0 # from 0 to 100
    accepted_transitions = ""
    rejected_transitions = ""     

    lesson_learned_description = ''' 
        Lesson learne: scheduling thread on thread ''' + str(i) + ''' 
        part 1 : core_''' + str(j) + '''_state = 0
        energy efficiency should increase
    '''
   
    command = '''
    SELECT
        avg(configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.energy_efficiency),
        avg(configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.configuration_id,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_OFF.configuration_id,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_OFF
                WHERE
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state == 0) 
                AS  configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_OFF, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.configuration_id,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state,
            configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_OFF.configuration_id,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state,
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_OFF
                WHERE
                    configuration_description_core_''' + str(j) + '''_OFF.core_''' + str(i) + '''_state == 1) 
                AS  configuration_description_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_OFF_core_''' + str(i) + '''_ON; '''



    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "0 -> 1" , 50,
                                                                             score, accepted_transitions, rejected_transitions)
    

    
    lesson_learned_description = '''
    lesson learned: increasing scheduling thread on thread ''' + str(i) + '''
        part 2: core_''' + str(j) + '''_state = 1
        energy efficiency should increase
    '''
    command = '''
    SELECT
        avg(configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.energy_efficiency),
        avg(configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.configuration_id,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.core_''' + str(i) + '''_state,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_ON.configuration_id,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_ON
                WHERE
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state == 0) 
                AS  configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_OFF, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.configuration_id,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.core_''' + str(i) + '''_state,
            configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON.core_''' + str(j) + '''_state
        FROM
                (SELECT
                    configuration_description_core_''' + str(j) + '''_ON.configuration_id,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state,
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(j) + '''_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_''' + str(i) + '''_state,
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_''' + str(j) + '''_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_''' + str(j) + '''_ON
                WHERE
                    configuration_description_core_''' + str(j) + '''_ON.core_''' + str(i) + '''_state == 1) 
                AS  configuration_description_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_''' + str(j) + '''_ON_core_''' + str(i) + '''_ON;  '''

    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]
   
    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "0 -> 1" , 50,
                                                                             score, accepted_transitions, rejected_transitions)
    
    

    validation_score = str(score) + " %" 
    # variable to increase, chipset state,  suitable-contraindicated-or-neutral, validation score, [accepted transition(s)], [rejected transition(s)]   
    return variable_to_increase + "," + chipset_state + "," + suitable_contraindication_or_neutral + ","  + \
                    validation_score + "," + accepted_transitions + "," + rejected_transitions

# this function has been generalized with the function above
def validate__scheduling_thread_on_core_0_no_matter_core_1_state( conn, expected_efficiency_behavior ): # conn is the connection handle to the data base
    variable_to_increase = "core 0 state" 
    suitable_contraindication_or_neutral = "suitable - efficiency should increase" if expected_efficiency_behavior == "increases" else "contraindicated -  efficiency should decrease" if expected_efficiency_behavior == "decreases" else "neutral - efficiency should be stable" 
    chipset_state = "Core 1 is ON or OFF"
    
    validation_score = "NULL" #(Validated - or Not Validated  [Score%])
    score = 0 # from 0 to 100
    accepted_transitions = ""
    rejected_transitions = ""     

    lesson_learned_description = ''' 
        Lesson learne: scheduling thread on thread 0
        part 1 : core_1_state = 0
        energy efficiency should increase
    '''
    command = '''
    SELECT
        avg(configuration_description_measurements_core_1_OFF_core_0_OFF.energy_efficiency),
        avg(configuration_description_measurements_core_1_OFF_core_0_ON.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_1_OFF_core_0_OFF.configuration_id,
            configuration_description_core_1_OFF_core_0_OFF.core_0_state,
            configuration_description_core_1_OFF_core_0_OFF.core_1_state
        FROM
                (SELECT
                    configuration_description_core_1_OFF.configuration_id,
                    configuration_description_core_1_OFF.core_0_state,
                    configuration_description_core_1_OFF.core_1_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_0_state,
                        configuration_description__google_pixel_4a_5g.core_1_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_1_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_1_OFF
                WHERE
                    configuration_description_core_1_OFF.core_0_state == 0) 
                AS  configuration_description_core_1_OFF_core_0_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_1_OFF_core_0_OFF, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_1_OFF_core_0_ON.configuration_id,
            configuration_description_core_1_OFF_core_0_ON.core_0_state,
            configuration_description_core_1_OFF_core_0_ON.core_1_state
        FROM
                (SELECT
                    configuration_description_core_1_OFF.configuration_id,
                    configuration_description_core_1_OFF.core_0_state,
                    configuration_description_core_1_OFF.core_1_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_0_state,
                        configuration_description__google_pixel_4a_5g.core_1_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_1_state == 0) /* see if we can reduce this */  
                    AS configuration_description_core_1_OFF
                WHERE
                    configuration_description_core_1_OFF.core_0_state == 1) 
                AS  configuration_description_core_1_OFF_core_0_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_1_OFF_core_0_ON; '''
    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "0 -> 1" , 50,
                                                                             score, accepted_transitions, rejected_transitions)
    

    
    lesson_learned_description = '''
    lesson learned: increasing scheduling thread on thread 0
        part 2: core_1_state = 1
        energy efficiency should increase
    '''
    command = '''
    SELECT
        avg(configuration_description_measurements_core_1_ON_core_0_OFF.energy_efficiency),
        avg(configuration_description_measurements_core_1_ON_core_0_ON.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_1_ON_core_0_OFF.configuration_id,
            configuration_description_core_1_ON_core_0_OFF.core_0_state,
            configuration_description_core_1_ON_core_0_OFF.core_1_state
        FROM
                (SELECT
                    configuration_description_core_1_ON.configuration_id,
                    configuration_description_core_1_ON.core_0_state,
                    configuration_description_core_1_ON.core_1_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_0_state,
                        configuration_description__google_pixel_4a_5g.core_1_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_1_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_1_ON
                WHERE
                    configuration_description_core_1_ON.core_0_state == 0) 
                AS  configuration_description_core_1_ON_core_0_OFF
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_1_ON_core_0_OFF, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_core_1_ON_core_0_ON.configuration_id,
            configuration_description_core_1_ON_core_0_ON.core_0_state,
            configuration_description_core_1_ON_core_0_ON.core_1_state
        FROM
                (SELECT
                    configuration_description_core_1_ON.configuration_id,
                    configuration_description_core_1_ON.core_0_state,
                    configuration_description_core_1_ON.core_1_state
                FROM
                    (SELECT
                        configuration_description__google_pixel_4a_5g.configuration_id,
                        configuration_description__google_pixel_4a_5g.core_0_state,
                        configuration_description__google_pixel_4a_5g.core_1_state
                    FROM
                        configuration_description__google_pixel_4a_5g
                    WHERE
                        configuration_description__google_pixel_4a_5g.core_1_state == 1) /* see if we can reduce this */  
                    AS configuration_description_core_1_ON
                WHERE
                    configuration_description_core_1_ON.core_0_state == 1) 
                AS  configuration_description_core_1_ON_core_0_ON
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_core_1_ON_core_0_ON;  '''

    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]
   
    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "1 -> 2" , 50,
                                                                             score, accepted_transitions, rejected_transitions)
    
    

    validation_score = str(score) + " %" 
    # variable to increase, chipset state,  suitable-contraindicated-or-neutral, validation score, [accepted transition(s)], [rejected transition(s)]   
    return variable_to_increase + "," + chipset_state + "," + suitable_contraindication_or_neutral + ","  + \
                    validation_score + "," + accepted_transitions + "," + rejected_transitions

def validate__increasing_little_sockect_freq_when_core_6_state_freq_level_is_3( conn, expected_efficiency_behavior): # conn is the connection handle to the data base
    
    variable_to_increase = "little socket frequency" 
    suitable_contraindication_or_neutral = "suitable - efficiency should increase" if expected_efficiency_behavior == "increases" else "contraindicated -  efficiency should decrease" if expected_efficiency_behavior == "decreases" else "neutral - efficiency should be stable" 
    chipset_state = "Medium frequency is high"
    
    validation_score = "NULL" #(Validated - or Not Validated  [Score%])
    score = 0 # from 0 to 100
    accepted_transitions = ""
    rejected_transitions = ""     

    lesson_learned_description = ''' 
        Lesson learne: increasing little_socket_frequency
        part 1: increasing little_socket_frequency from 0 to 1
        core_6_state_freq_level = 3
        energy efficiency should decreases
    '''
    command = '''
    SELECT
        avg(configuration_description_measurements_littel_freq_0_Medium_socket_freq_H.energy_efficiency),
        avg(configuration_description_measurements_littel_freq_1_Medium_socket_freq_H.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
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
        USING(configuration_id))  
        AS configuration_description_measurements_littel_freq_0_Medium_socket_freq_H, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
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
        USING(configuration_id))  
        AS configuration_description_measurements_littel_freq_1_Medium_socket_freq_H; '''
    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]

    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "0 -> 1" , 50,
                                                                             score, accepted_transitions, rejected_transitions)
    

    
    

    lesson_learned_description = '''
    lesson learned: increasing little_socket_frequency
        part 2:  increasing little_socket_frequency from 1 to 2
        core_6_state_freq_level = 3
        energy efficiency should decreases
    '''
    command = '''
    SELECT
        avg(configuration_description_measurements_littel_freq_1_Medium_socket_freq_H.energy_efficiency),
        avg(configuration_description_measurements_littel_freq_2_Medium_socket_freq_H.energy_efficiency)
    FROM
        /*initial values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
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
        USING(configuration_id))  
        AS configuration_description_measurements_littel_freq_1_Medium_socket_freq_H, 
        
        /*end values*/     
        (SELECT
            configuration_measurements.energy_efficiency,
            configuration_description_little_freq_2_Medium_socket_freq_H.configuration_id,
            configuration_description_little_freq_2_Medium_socket_freq_H.little_socket_frequency,
            configuration_description_little_freq_2_Medium_socket_freq_H.core_6_state_freq_level
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
                    configuration_description_Medium_socket_freq_H.little_socket_frequency == 2) 
                AS  configuration_description_little_freq_2_Medium_socket_freq_H
        INNER JOIN 
                configuration_measurements 
        USING(configuration_id))  
        AS configuration_description_measurements_littel_freq_2_Medium_socket_freq_H; '''

    cursor = conn.execute(command)
    print (lesson_learned_description)
    variation = 0
    for row in cursor:
        print("before = ", row[0])
        print("after = ", row[1])
        print("variation ", row[1] - row[0])
        variation =  row[1] - row[0]
   
    score, accepted_transitions, rejected_transitions = compute_decision_score(expected_efficiency_behavior, variation, "1 -> 2" , 50,
                                                                             score, accepted_transitions, rejected_transitions)
    
    

    validation_score = str(score) + " %" 
    # variable to increase, chipset state,  suitable-contraindicated-or-neutral, validation score, [accepted transition(s)], [rejected transition(s)]   
    return variable_to_increase + "," + chipset_state + "," + suitable_contraindication_or_neutral + ","  + \
                    validation_score + "," + accepted_transitions + "," + rejected_transitions
     
    """

    if False:
        # Validation of the ability of the kernel ridge to classify configurations. 
        # Intersection between first N configurations in measurement and first N  of the same configurations when estimated by the model.   
        command = '''
        SELECT COUNT(*) FROM
            ((SELECT * FROM configuration_measurements ORDER BY  energy_efficiency  DESC LIMIT 10) AS configuration_measurements_ordered
            INNER JOIN 
            (SELECT * FROM
                (SELECT * FROM  configuration_efficiency_estimation INNER JOIN configuration_measurements USING(configuration_id))
            ORDER BY  energy_efficiency  DESC LIMIT 10) AS configuration_estimation_of_the_one_mesured__ordered
            USING(configuration_id))
        INNER JOIN
            configuration_representations
        USING(configuration_id)
        '''
        #command = '''SELECT * FROM configuration_description__google_pixel_4a_5g INNER JOIN configuration_measurements USING(configuration_id);''' 
        cursor = conn.execute(command)
        print ("--- intersection result joint result")
        for row in cursor:
            print("before = ", row[0])
    """        
        
