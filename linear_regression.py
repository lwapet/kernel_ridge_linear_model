from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import utils_functions as utils
import fill_data as data
import statsmodels.api as sm


####################################  Prediction on google pixel 4a 5g
X_user_friendly = data.X_user_friendly_google_pixel_4a_5g
print ("*** Total configurations user friendly: ", X_user_friendly)
X = data.X_google_pixel_4a_5g
print ("*** Total Configurations formatted: ", X)
X_dict = data.X_dict_google_pixel_4a_5g
print ("*** Total Configurations dictionnary: ", X_dict)
y = data.y_google_pixel_4a_5g
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
"""
reg =  LinearRegression()
reg.fit(X_train, y_train)
reg_y_test = reg.predict(X_test)
"""
############# Using stats_models
ols = sm.OLS(y_train, X_train)
reg = ols.fit()
reg_y_test = reg.predict(X_test)
print("Predicted y test = ", reg_y_test)


_, (orig_data_ax, testin_data_ax, linear_regression_ax) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 13))
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

linear_regression_ax.bar(X_user_friendly_test,reg_y_test, width=0.4)
# Add title and axis names
linear_regression_ax.set_title('Predited energy/workload ratio')
linear_regression_ax.set_xlabel('Configuration')
linear_regression_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
linear_regression_ax.tick_params(size=8)

#_ = linear_regression_ax.set_title("Predicted data\n using linear regression, R2 = " + str(reg.score(X_test, y_test)))
_ = linear_regression_ax.set_title("Predicted data\n using linear regression")


#print("error = ", reg.score(X_test, y_test))
#print("parrams = " ,  reg.get_params(True))
print(reg.summary())

plt.gcf().autofmt_xdate()
plt.xticks(fontsize=8)
plt.savefig("linear_regression_prediction_on_google_pixel_4a_5g.png")
plt.clf()
plt.cla()
plt.close()


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
reg =  LinearRegression()
reg.fit(X_train, y_train)
reg_y_test = reg.predict(X_test)
############# 


_, (orig_data_ax, testin_data_ax, linear_regression_ax) = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 13))
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

linear_regression_ax.bar(X_user_friendly_test,reg_y_test, width=0.4)
# Add title and axis names
linear_regression_ax.set_title('Predited energy/workload ratio')
linear_regression_ax.set_xlabel('Configuration')
linear_regression_ax.set_ylabel(r'Energy consumed/Workload ($\times 10E-11$)')
linear_regression_ax.tick_params(size=8)

_ = linear_regression_ax.set_title("Predicted data\n using linear regression, R2 = " + str(reg.score(X_test, y_test)))

print("error = ", reg.score(X_test, y_test))
print("parrams = " ,  reg.get_params(True))
print("regressors = " ,  reg.coef_)
plt.gcf().autofmt_xdate()
plt.xticks(fontsize=8)
plt.savefig("linear_regression_prediction_on_samsung_galaxy_s8.png")
plt.clf()
plt.cla()
plt.close()
"""

"""
## original code version
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
>>> # y = 1 * x_0 + 2 * x_1 + 3
>>> y = np.dot(X, np.array([1, 2])) + 3
>>> reg = LinearRegression().fit(X, y)
>>> reg.score(X, y)
"""