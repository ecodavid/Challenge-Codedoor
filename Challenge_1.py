#!/usr/bin/env python
# coding: utf-8

# # Task 1: List vs. NumPy Array

# #### Implementation: Find length of my_list

# In[3]:


my_list = [62, 53, 69, 84, 83, 80, 98, 74, 70, 92, 68, 71, 54, 87, 85, 88, 63, 61, 66, 99]

# TODO: Get length of a list

def list_length(input_list):
    return len(input_list)
    return None

# Printing the length of the list
print(list_length(my_list))


# #### Implemention: Find min and max of list

# In[4]:


import random

my_list = [62, 53, 69, 84, 83, 80, 98, 74, 70, 92, 68, 71, 54, 87, 85, 88, 63, 61, 66, 99]

#TODO: Find min and max of a list
def list_max_min(input_list): 
    
    minNumber = min(input_list)
    maxNumber = max(input_list)
    return [minNumber,maxNumber]

# Printing min and max of random numbers in a list
result = list_max_min(my_list)
print("Minimum number  {}".format(result[0]))
print("Maximum number: {}".format(result[1]))


# #### Arithmetic and Statistical Operations on lists

# In[5]:


# TODO: multiply each element of the list with the given coeff

coeff = 7.231671067280546 # The constant to be multiplied to each element of the list 

res_list = [i * coeff for i in my_list] # List comprehension

res_list = res_list

# Print the result of the multiplication 
print(res_list)

# TODO: take the average of the elements of the list
mean_res_list = sum(res_list)/len(res_list)

mean_res_list = mean_res_list

# Showing the result of taking the average
print(mean_res_list)


# #### Working with NumPy Arrays

# In[6]:


# First element is the result of multiplication of the array with the coefficient coeff. 
# This element is a NumPy array.
# Second element is a list with a length of 2 elements with this format: [min_array, max_array]
# For the third element: take the average of the elements in NumPy array stored in the first 
# element. This third element is a float number.

# Import NumPy library necessary for this part 
import numpy as np

my_array = np.array(my_list) # The NumPy array format of the list is obtained

def math_array(input_array):
    coeff = 7.231671067280546 # The constant to be multiplied 
    multiplication_array = my_array * coeff
    Min_Max_array =  multiplication_array.min(),multiplication_array.max()
    Avg_array = multiplication_array.mean()
   
    res = [multiplication_array, Min_Max_array, Avg_array]
    return res 

# Displaying the result of calling the implemented function on my_array
print(math_array(my_array))


# #### NumPy Arrays and files

# In[7]:


# Change requested by tutor

import pandas as pd
import os
print(os.getcwd())
grades_np = pd.read_csv(r"C:\Users\Ivan Rojas\Documents\GitHub\ds-ml-ai-challenge-ecodavid\data\grades.csv", header = None)

# Print the grades

display (grades_np)

number_of_grades = grades_np.size
# number_of_grades = None # A variable for taking the size of the array

# Display the size of the array
print("The number of grades are: " + str(number_of_grades))


# #### Boolean Indexing

# In[8]:


# TODO: Split the NumPy array 'grades_np' into even and odd parts

grades_np = np.array(grades_np)

even_grades = grades_np[grades_np%2==0] #Select an array with elements divisable by 2 and with
                                        #remaining equal to cero
odd_grades = grades_np[grades_np%2!=0] 
            
even_grades = even_grades
odd_grades = odd_grades

even_grades_size = even_grades.size
odd_grades_size = odd_grades.size

# even_grades_size = even_grades_size
# odd_grades_size = odd_grades_size

# Display the resulted arrays from the splitting process
print("The even grades are: "+ str(even_grades))
print("The odd grades are: "+ str(odd_grades))

# Display the size of the arrays
print("The number of even grades is: " + str(even_grades_size))
print("The number of odd grades is: " + str(odd_grades_size))


# #### Arithmetic Operations and Mask Indexing

# In[9]:


# TODO: Take the average of the even grades and specify the indexes with the elements larger 
# than the mean

mean_even_grades = even_grades.mean()# The variable for holding the mean value of the even grades
even_grades > mean_even_grades
greater_than_mean_inds = np.nonzero(even_grades > mean_even_grades)

# Display the taken mean
print("The mean value of the even grades is: " + str(mean_even_grades))

greater_than_mean_inds = greater_than_mean_inds # A list of indexes with the values larger than the "mean_even_grades"

# Displaying the indices
print("Indices with the values greater than the mean are: " + str(greater_than_mean_inds))


# #### Sorting, Arithmetic and Statistical Operations on NumPy arrays

# In[10]:


# Sort the array of odd_grades from the earlier section in an ascending order.
# Do the same for the even_grade with the reverse order. 
# Try to get the median of the odd_grades
# Then calculate the difference of the median and mean of odd_grades.
# Calculate the Standard Deviation of odd_grades.

# TODO: Sort "odd_grades" in an ascending and "even_grades" in a descending order

sorted_odd_grades = np.sort(odd_grades)
sorted_odd_grades = sorted_odd_grades 

reverse_sorted_even_grades = -np.sort(-even_grades)
reverse_sorted_even_grades = reverse_sorted_even_grades

# Printing the sorted arrays:
print(sorted_odd_grades)
print(reverse_sorted_even_grades)

# Variables which hold median, mean and the absolute difference of median and mean
median_odd_grades = np.median(odd_grades)
mean_odd_grades = np.mean(odd_grades)
mean_median_diff = np.abs(median_odd_grades - mean_odd_grades)
std_odd_grades = np.std(odd_grades)

# Display median, mean, difference of them
print("The median of the odd grades is: " + str(median_odd_grades))
print("The mean of the odd grades is: " + str(mean_odd_grades))
print("The difference of median and the mean of the odd grades is: " + str(mean_median_diff))
print("The standard deviation for odd grades is: " + str(std_odd_grades))


# #### Multi-Dimensional NumPy Arrays

# In[11]:


# TODO: Split the "grades_np" into an array with 4 rows, take the mean, median, std of each 
# row. 

splited_grades_np = grades_np.reshape(4,25)

means = np.mean(splited_grades_np, axis = 1)
medians = np.median(splited_grades_np, axis = 1)
stds = np.std(splited_grades_np, axis = 1)

mean_median_diff_ind = np.argmax(abs(means - medians)) 
print(mean_median_diff_ind) # 2 is the index of the row with the largest difference between 
                            # mean and median

# Variables for holding the mean, median, and std
means = means
medians = medians
stds = stds
mean_median_diff_ind = mean_median_diff_ind 
                                           

# Display the taken values
print("mean values: " + str(means))
print("median values: " + str(medians))
print("std values: " + str(stds))

# The NumPy arrays for holding the new splited array with the shape of (4,:), 
# the sorted grades for each row, and the top 3 grades for each row

sorted_splited_grades = np.sort(splited_grades_np)
best_grades = sorted_splited_grades[ :,-3:] # This range contains all rows and selected
                                            # the last 3 columns of the sorted array

# Display the Best score
print(best_grades)


# # Task 2. Data Visualization

# In[12]:


# Visualization
import matplotlib.pyplot as plt


# #### Bar Plot

# In[13]:


# take a look at  xticks. Using this function, you can assign each 2 students for instance a 
# tick.

Student_index = np.arange(len(even_grades))
plt.bar(Student_index, even_grades)
plt.xlabel('Student Index')
plt.ylabel('grade')
plt.title('Student even grades')
plt.xticks(np.arange(1 ,46, step=2))


# #### Histogram

# In[14]:


# Generate a histogram of the even_grades with 9 bins on the horizontal axis

plt.hist(even_grades, bins = 9, rwidth = 0.95, color = 'green') 
plt.xlabel('grade bins')
plt.ylabel('num of students')
plt.title('grades histogram')


# #### Subplots (Bonus):

# In[15]:


class_0 = even_grades[0:8]
class_1 = even_grades[8:16]
x = np.arange(len(class_0))
width = 0.35
fig, ax = plt.subplots()
c0 = ax.bar(x - width/2, class_0, width, color = 'orange', label = 'setA')
c1 = ax.bar(x + width/2, class_1, width, color = 'blue', label = 'setB')
ax.set_xlabel('set - class')
ax.set_ylabel('grade')
ax.set_title('Sample grades in two sets')
ax.legend()
print(class_0)
print(class_1)


# # Task 3: Pandas Dataframe

# #### Getting Started

# In[16]:


import pandas as pd

input_df = pd.read_csv('Ecommerce-Customers.csv')
input_df.head(5)


# In[17]:


# TODO: Get the shape of the input_df

# Variable for holding the shape
input_shape = [None]

input_shape = input_df.shape
print(input_shape)
input_df.dtypes
# Display the name of the columns and the type of the values


# In[18]:


# TODO: Display top 5 rows of input_df

input_df.head(5)


# #### Average Time spent on Website and App

# In[19]:


# TODO: Write function the mean and std of app and website column

def df_math():
    app_mean = input_df['Time on App'].mean()
    app_std = input_df['Time on App'].std()
    web_mean = input_df['Time on Website'].mean()
    web_std = input_df['Time on Website'].std()
    res = [[app_mean, app_std], [web_mean, web_std]] # Format: [[app_mean, app_std], [web_mean, web_std]]
    
    return res 

# Display the return value of the function
print(df_math())


# #### Finding max and the max_ID

# In[20]:


def max_finder(input_DataFrame):
     
    max_time_app = input_df['Time on App'].max()
    email_app = input_df.loc[input_df['Time on App'].idxmax(),'Email'] 
    max_time_web = input_df['Time on Website'].max()
    email_web = input_df.loc[input_df['Time on Website'].idxmax(),'Email']
     
    return [email_app, max_time_app, email_web, max_time_web] # format explained above
    
result = max_finder(input_df)

# Displaying the result
print("Email address: {}".format(result[0]))
print("Max time spent on the app: {}".format(result[1]))
print("Email address: {}".format(result[2]))
print("Max time spent on the web: {}".format(result[3]))


# #### Sort and Selection

# In[21]:


def best_score_finder(input_DataFrame):# Function for getting the costumers with the best scores
    scr = input_df['Avg Session Length']*input_df['Length of Membership']# new score formula
    input_df['scr']= scr # Assign the score to a new column called 'scr'
    # sort the users based on score value and select the top 5 users with the highest score.
    best_scores = input_df.sort_values('scr', ascending = False).head(5) 
    return best_scores

# Displaying the results of function with input_df as input
best_scores = best_score_finder(input_df) # input_df
display(best_scores)


# In[22]:


# Sort input_df using the values in the column "Yearly Amount Spent". Return the top 5 rows. 
# Store the results of the function in a variable called “best_yearly_spent”.

def best_yearly_spent_finder():
    best_yearly_spent = input_df.sort_values('Yearly Amount Spent', ascending = False).head(5)
    return  best_yearly_spent # Return value is of type "panda dataframe" with 5 rows

best_yearly_spent = best_yearly_spent_finder() # input_df 
display(best_yearly_spent)


# #### Finding Common Elements

# In[23]:


# Find the intersection of the two dataframes. We want to find the email addresses 
# of the users that exist in both dataframes.

def comparisor(a,b):
    comparisor = pd.merge(best_scores, best_yearly_spent, on = 'Email', how = 'inner')['Email']

    return comparisor  # a dataframe that contains the rows that exist in both dataframes a,b

comparisor(best_scores, best_yearly_spent)


# # Task 4. Regression

# In[24]:


# Select the numerical features and formed a new dataframe "num_df". In the cell below, try
# to form this dataframe.

# TODO: Selecting the numerical columns of "input_df"

num_df = input_df.select_dtypes(include = ['float64', 'int'])
num_df = [num_df]
# Print the result
display(num_df)


# In[25]:


# Develop a linear regression model and return the mean squared error on the test set 

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error

def predict_model(train_set, test_set):
     
    test_set = pd.read_csv('test.csv')
    train_set = pd.read_csv('train.csv')
    
    y_train = train_set['Yearly Amount Spent']
    y_test = test_set['Yearly Amount Spent']
    
    x_train = train_set[['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    x_test = test_set[['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    
    linreg = LinearRegression()
    linreg.fit(x_train,y_train)
    
    l_model = linreg.predict(x_test)
    
    res = mean_squared_error(y_test, l_model)
    
    return res

err = predict_model(train_set, test_set)

print("The mean square error of the model on the test set is: {}".format(err))

