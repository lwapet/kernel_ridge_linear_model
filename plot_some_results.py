import matplotlib.pyplot as plt
import numpy as np
import math
import utils_functions as utils
import random
automatization_summaries_folder = "/mnt/c/Users/lavoi/opportunist_task_on_android/scripts_valuable_files/experiment_automatization/summary_files_only"
output_data_folder = "some_input_data_plots/"
number_of_data_to_plot = 40



def select_only_some_configurations(number_of_combinaison, configurations, phone_energy, phone_power, workload, energy_efficiency):
    # This function select only number_of_combinaisons to plot from the datas passed as parameters 
    # The datas passed as parameters are configurations, phone_energy, phone_power, workload
    #
    # 
    print("--- Computing combinations with naive approach")

    list_of_retained_indices=random.sample(range(0, len(configurations)), number_of_combinaison)
    configurations_retained=[]
    phone_energy_retained = []
    phone_power_retained = []
    workload_retained = []
    energy_efficiency_retained= []
   
    for indice_retained in list_of_retained_indices: 
        configurations_retained.append(configurations[indice_retained])
        phone_energy_retained.append(phone_energy[indice_retained])
        phone_power_retained.append(phone_power[indice_retained])
        workload_retained.append(workload[indice_retained])
        energy_efficiency_retained.append(energy_efficiency[indice_retained])
        
    return configurations_retained, phone_energy_retained, phone_power_retained, workload_retained, energy_efficiency_retained





configurations = utils.get_data_from_summary_folder(automatization_summaries_folder, "configurations", "human_readable_format" )
print ("*** Total configurations in user friendly format: ", configurations)
phone_energy = utils.get_data_from_summary_folder(automatization_summaries_folder, "energy")
print ("*** Total phone energy: ", phone_energy)
phone_power = utils.get_data_from_summary_folder(automatization_summaries_folder, "power")
print ("*** Total phone power: ", phone_energy)
workload = utils.get_data_from_summary_folder(automatization_summaries_folder, "workload")
print ("*** Total workloads: ", workload)
energy_efficiency = utils.get_data_from_summary_folder(automatization_summaries_folder, "energy_efficiency")
print ("*** Total energy efficiency: ", energy_efficiency)

configurations, phone_energy, phone_power, workload, energy_efficiency = select_only_some_configurations(number_of_data_to_plot ,configurations, phone_energy, phone_power, workload, energy_efficiency)
print ("Size of Configurations list: ", len(configurations))


#configurations = ["mouse",  "idle"       ,"1-0",  "2-0",  "3-0",       "4-0", "5-0", "6-0",              "6-1",    "6-2"             ,"0-1-0",  "0-0-1"   ,"0-2",              "1-1",  "2-1", "3-1",      "1-2" ]                                                        
#phone_energy= [ 1308, 11047.73,       29197.24, 38598.14, 46077.78,    53629.51, 58756.40, 66053.62,     106210.78 , 123823.21,       66877.5,  77632.30,    115136.69,        73048.75, 79000.24 , 92143.93     , 96571.58 ]                                                                                                   
#phone_power = [ 40.16, 336.07,        879.245 , 1146.12,  1357.74  ,   1572.76, 1718.15, 1918.21,         2970.56 ,  3397.90,          1938.45, 2214.51,  3179.93,        2106.08,    2267.14 , 2614.01   , 2717.37]                  
#workload  = [ 0.1,         0.9,        1.60362,    3.368,  4.9563   ,     6.906,  7.3213, 10.85176,          17.08829 ,  21.384681,       6.7442, 7.64552224, 13.210451922,     8.47040,  10.3032,    12.38267        , 12.72335]          



fig = plt.figure()

width = 0.35  
plt.bar(configurations,phone_power, width, label='With new usb cable')

# Add title and axis names
plt.title('Avg power used by the phone according to the configuration \n BM = Battery at middle level (50%)')
plt.xlabel('Number of CPUs')
plt.ylabel('AVG Power')

#plt.xticks(fontsize=8)
fig.autofmt_xdate()

plt.savefig(output_data_folder + "Power_absorbed_according_to_configuration_Google_pixel_.png")
plt.legend(loc='upper left')
plt.clf()
plt.cla()
plt.close()
########################################################################################

fig = plt.figure()
plt.bar(configurations,phone_energy, width=0.4)
 
# Add title and axis names
plt.title('Energy consumed according to the \n number of configuration')
plt.xlabel('Number of CPUs')
plt.ylabel('Battery cpu usage')
#plt.xticks(fontsize=8)
fig.autofmt_xdate()
plt.savefig(output_data_folder + "Energy_usage_according_to_configuration_Google_pixel.png")

plt.clf()
plt.cla()
plt.close()

#################################################

fig = plt.figure()
plt.bar(configurations,workload, width=0.4)

# Add title and axis names
plt.title('New computed Workload according to the configuration')
plt.xlabel('Number of CPUs')
plt.ylabel(r'Workload ($\times 10E11$)')
#plt.xticks(fontsize=8)
fig.autofmt_xdate()
plt.savefig(output_data_folder + "Workload_according_to_configuration_google_pixel_w.png")

plt.clf()
plt.cla()
plt.close()
################################################

fig = plt.figure()
plt.bar(configurations,energy_efficiency, width=0.4)
# Add title and axis names
plt.title('Energy efficiency Energy/Workload according to \n the configuration (lower is better)')
plt.xlabel('Number of CPUs')
plt.ylabel(r'Battery cpu/Workload ($\times 10E-11$)')
#plt.xticks(fontsize=8)
fig.autofmt_xdate()
plt.savefig(output_data_folder + "Ratio_energy_by_workload_according_to_configuration_Google_pixel.png")

plt.clf()
plt.cla()
plt.close()


print ("**** Total configurations in user friendly format: ", configurations)
print ("*** Total phone energy: ", phone_energy)
print ("*** Total phone power: ", phone_power)
print ("*** Total workloads: ", workload)
print ("*** Total energy efficiency: ", energy_efficiency)
################################################

print ("printing plots")
_, (power_ax, energy_ax, workload_ax, energy_efficiency_ax) = plt.subplots(nrows=4, figsize=(20, 20))

power_ax.bar(configurations,phone_power, width=0.4)
# Add title and axis names
power_ax.set_title('Power absorbed')
power_ax.set_xlabel('Configuration')
power_ax.set_ylabel(r'Power absorbed')
power_ax.tick_params(size=8)


energy_ax.bar(configurations,phone_energy, width=0.4)
# Add title and axis names
energy_ax.set_title('Energy consumed')
energy_ax.set_xlabel('Configuration')
energy_ax.set_ylabel(r'Energy (mAh)')
energy_ax.tick_params(size=8)


workload_ax.bar(configurations,workload, width=0.4)
# Add title and axis names
workload_ax.set_title('Workload')
workload_ax.set_xlabel('Configuration')
workload_ax.set_ylabel ('Workload')#(r'Workload ($\times 10E-11$)')
workload_ax.tick_params(size=8)

energy_efficiency_ax.bar(configurations,energy_efficiency, width=0.4)
# Add title and axis names
energy_efficiency_ax.set_title('Energy efficiency')
energy_efficiency_ax.set_xlabel('Configuration')
energy_efficiency_ax.set_ylabel ('Energy efficiency')#(r'Energy efficiency ($\times 10E-11$)')
energy_efficiency_ax.tick_params(size=8)


_ = energy_efficiency_ax.set_title("Some data obtained from experiment automatization, number of data: " + repr(len(configurations)))



plt.gcf().autofmt_xdate()
plt.xticks(fontsize=8)
plt.savefig(output_data_folder + "some_data_from_automatized_experiments.png")
plt.clf()
plt.cla()
plt.close()