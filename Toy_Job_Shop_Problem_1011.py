#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model
import collections
from datetime import timedelta
import datetime
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np


# # Define the model

# In[2]:


def MinimalJobshopSat():
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()


# In[3]:


model = cp_model.CpModel()


# # Import Data

# In[5]:


imported_data_df = pd.read_excel('Toy Job Shop problem.xlsx', sheet_name='Data')

#print (imported_data_df)

#check column format
imported_data_df.dtypes

#Organisation                      object
#Task 1 completion date    datetime64[ns]
#Task 2 Completion date    datetime64[ns]
#Task 3 length                      int64
#Task 4 length                      int64
#dtype: object

initial_date = '2019-10-01'

#Create initialisation date 2019-10-01
imported_data_df=imported_data_df.assign(Initialisation_Date = pd.to_datetime(initial_date))

#create Task_1_length column (int)
imported_data_df=imported_data_df.assign(Task_1_length = ((imported_data_df['Task 1 completion date'] - imported_data_df['Initialisation_Date'])/np.timedelta64(1,'D')).astype(int))

#creating Task_2_lenght on the assumption that task two can start only after task 1 is completed. is that the case?
imported_data_df=imported_data_df.assign(Task_2_length = ((imported_data_df['Task 2 Completion date'] - imported_data_df['Task 1 completion date'])/np.timedelta64(1,'D')).astype(int))
#df['diff_days'] = df['End_date'] - df['Start_date']
#df['diff_days']=df['diff_days']/np.timedelta64(1,'D')

#find max(Task 2 Completion date) and define Transition_Date_Start as max(Task 2 Completion date) +5
#imported_data_df=imported_data_df.assign(Max_Preparation_Task_2_dt = max(imported_data_df['Task 2 Completion date']))
#imported_data_df=imported_data_df.assign(Transition_Date_Start =  imported_data_df['Max_Preparation_Task_2_dt'] + 5 )
#imported_data_df["Transition_Date_Start"] = imported_data_df["Max_Preparation_Task_2_dt"] + timedelta(days=5)

#define the end date of task2 completion for org 1 to 5, which is the lattest of the task 2 completion data plus another 5 days
org_1_5_task2complete_date = max(imported_data_df.loc[0:4,'Task 2 Completion date']) + timedelta(days=5)
#define the latter of the org6 and 7 task 2 completion date
org_6_7_task2complete_date = max(imported_data_df.loc[5:6,'Task 2 Completion date'])
#define transition start date for org 1 to 5, they can only start after 6 and 7 are finished with their task 2
org_1_5_transition_Date_Start = max(org_1_5_task2complete_date,org_6_7_task2complete_date)
#insert transition start date into the dataframe
imported_data_df["Transition_Date_Start"] = org_1_5_transition_Date_Start
#modify the transition start date for org 6 and 7
imported_data_df.loc[5:6,"Transition_Date_Start"] = imported_data_df.loc[5:6,'Task 2 Completion date']

#define Org2 Transition phase start
org_2_task3_startdate = imported_data_df.loc[1,"Transition_Date_Start"]
#define Org_4 Transition phase start
org_4_task3_startdate = imported_data_df.loc[3,"Transition_Date_Start"]
#define Org_5 start date for Task 3
org_5_startdate = min(org_2_task3_startdate,org_4_task3_startdate) + timedelta(days=7)
#insert Org_5 start date into dataframe
imported_data_df.loc[4,"Transition_Date_Start"] = org_5_startdate

#compute waiting days between task2 end date and transition start date 
wait_days = imported_data_df['Transition_Date_Start'] - imported_data_df['Task 2 Completion date']
#convert the days into integer format
Pause_length=[]
for i in wait_days:
    j = i.days
    Pause_length.append(j)
#Add this column to the dataframe 
imported_data_df['Pause_length'] = Pause_length

imported_data_df.head()


# # Define the data

# In[6]:



#task_1 = [(df.iloc[i, 0], df.iloc[i, 6]) for i in range(0,5)]
#task_1 = [([int(s) for s in t.split() if s.isdigit()][0],n) for (t,n) in task_1]
#task_2 = [(df.iloc[i, 0], df.iloc[i, 7]) for i in range(0,5)]
#task_2 = [([int(s) for s in t.split() if s.isdigit()][0],n) for (t,n) in task_2]
#task_3 = [(df.iloc[i, 0], df.iloc[i, 3]) for i in range(0,5)]
#task_3 = [([int(s) for s in t.split() if s.isdigit()][0],n) for (t,n) in task_3]
#task_4 = [(df.iloc[i, 0], df.iloc[i, 4]) for i in range(0,5)]
#task_4 = [([int(s) for s in t.split() if s.isdigit()][0],n) for (t,n) in task_4]
#task_1 = [inst(s) for s in str.split() if s.isdigit() for (s,n) in task_1]

#jobs_data = [ # task = (machine_id, processing_time).
 #           task_1, task_2, task_3, task_4
  #          ]

# jobs_data = [  # task = (machine_id, processing_time).
#         [(0, 3), (1, 2), (2, 2)],  # Job0
#         [(0, 2), (2, 1), (1, 4)],  # Job1
#         [(1, 4), (2, 3)]  # Job2
#     ]

#wait_period = 5

#df=imported_data_df
df = imported_data_df[['Organisation','Task_1_length', 'Task_2_length','Pause_length','Task 3 length','Task 4 length']]

jobs_data = []
for j in range (1, len(df.columns)):
    ls = []
    for i in range (0,len(df.index)):
        task = (i+1, df.iloc[i,j])
        ls.append(task)
    jobs_data.append(ls)
    
org_count = 1 + max(task[0] for job in jobs_data for task in job)
all_organisations = range(org_count)


# In[7]:


#[print(job) for job in jobs_data]
print(jobs_data)


# # Define the variables

# In[8]:


# Named tuple to store information about created variables.
task_type = collections.namedtuple('task_type', 'start end interval')
# Named tuple to manipulate solution information.
assigned_task_type = collections.namedtuple('assigned_task_type',
                                            'start job index duration')


# In[9]:


# Create job intervals and add to the corresponding machine lists.
all_tasks = {}
org_to_intervals = collections.defaultdict(list)

# Computes horizon dynamically as the sum of all durations.
horizon = sum(task[1] for job in jobs_data for task in job)

for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            organisation = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, int(horizon), 'start' + suffix)
            end_var = model.NewIntVar(0, int(horizon), 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            org_to_intervals[organisation].append(interval_var)


# # Define the constraints

# In[10]:


# Create and add disjunctive constraints.
for organisation in all_organisations:
    model.AddNoOverlap(org_to_intervals[organisation])
    
# Precedences inside a job.
#for job_id, job in enumerate(jobs_data):
#    for task_id in range(len(job) - 1):
#        model.Add(all_tasks[job_id, task_id +
#                            1].start >= all_tasks[job_id, task_id].end)


# # Define the objective

# In[11]:


# Makespan objective.
obj_var = model.NewIntVar(0, int(horizon), 'makespan')
model.AddMaxEquality(obj_var, [
    all_tasks[job_id, len(job) - 1].end
    for job_id, job in enumerate(jobs_data)
])
model.Minimize(obj_var)


# # Declare the solver

# In[12]:


# Solve model.
solver = cp_model.CpSolver()
status = solver.Solve(model)


# # Display the results

# In[13]:


# Create one list of assigned tasks per machine.
assigned_jobs = collections.defaultdict(list)
for job_id, job in enumerate(jobs_data):
    for task_id, task in enumerate(job):
        organisation = task[0]
        assigned_jobs[organisation].append(
            assigned_task_type(
                start=solver.Value(all_tasks[job_id, task_id].start),
                job=job_id,
                index=task_id,
                duration=task[1]))

# Create per organisation output lines.
output = ''
for organisation in all_organisations:
    # Sort by starting time.
    assigned_jobs[organisation].sort()
    sol_line_tasks = 'Organisation ' + str(organisation) + ': '
    sol_line = '           '

    for assigned_task in assigned_jobs[organisation]:
        name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
        # Add spaces to output to align columns.
        sol_line_tasks += '%-10s' % name

        start = assigned_task.start
        duration = assigned_task.duration
        sol_tmp = '[%i,%i]' % (start, start + duration)
        # Add spaces to output to align columns.
        sol_line += '%-10s' % sol_tmp

    sol_line += '\n'
    sol_line_tasks += '\n'
    output += sol_line_tasks
    output += sol_line

# Finally print the solution found.
print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
print(output)


# Create per organisation output lines in date format
output_date = ''
for organisation in all_organisations:
    # Sort by starting time.
    assigned_jobs[organisation].sort()
    sol_line_tasks = 'Organisation ' + str(organisation) + ': '
    sol_line = '           '

    for assigned_task in assigned_jobs[organisation]:
        name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
        # Add spaces to output to align columns.
        sol_line_tasks += '%-10s' % name
        
        start_date = '2019/10/01'
        Initialisation_Date = datetime.datetime.strptime(start_date, '%Y/%m/%d')
        
        start = int(assigned_task.start)
        duration = int(assigned_task.duration)
        start_date = Initialisation_Date + timedelta(start)
        start_duration_date = Initialisation_Date + timedelta(start) + timedelta(duration)
        start_date_str = str(start_date)
        start_duration_date_str = str(start_duration_date)
        sol_tmp = '[%s,%s]' % (start_date_str, start_duration_date_str)
        # Add spaces to output to align columns.
        sol_line += '%-10s' % sol_tmp
        
    sol_line += '\n'
    sol_line_tasks += '\n'
    output_date += sol_line_tasks
    output_date += sol_line

# Finally print the solution found in datetime format.
print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
print(output)

# In[ ]:





# # Visualize the results

# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Declaring a figure "gnt" 
fig, gnt = plt.subplots() 

# Setting Y-axis limits 
gnt.set_ylim(0, 50) 

# Setting X-axis limits 
gnt.set_xlim(0, 160) 

# Setting labels for x-axis and y-axis 
gnt.set_xlabel('Date') 
gnt.set_ylabel('Organisation') 

# Setting ticks on y-axis 
gnt.set_yticks([15, 25, 35]) 
# Labelling tickes of y-axis 
gnt.set_yticklabels(['1', '2', '3']) 

# Setting graph attribute 
gnt.grid(True) 

# Declaring a bar in schedule 
gnt.broken_barh([(40, 20)], (30, 9),
facecolors =('tab:orange')) 

# Declaring multiple bars in at same level and same width 
gnt.broken_barh([(110, 10), (150, 10)], (10, 9),
facecolors ='tab:blue') 

gnt.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),
facecolors =('tab:red')) 

#plt.savefig("gantt1.png") 


# # Complete Code (Modularization)

# In[20]:


import pandas as pd
# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model
import collections
from datetime import timedelta
import datetime
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import matplotlib.pyplot as plt

def MinimalJobshopToy():
    #define the model
    model = cp_model.CpModel()
    
    '''Import Data'''
    #set data directory
    data_dir='Toy Job Shop problem.xlsx'
    
    imported_data_df = pd.read_excel(data_dir, sheet_name='Data')

    #print (imported_data_df)

    #check column format
    imported_data_df.dtypes

    #Organisation                      object
    #Task 1 completion date    datetime64[ns]
    #Task 2 Completion date    datetime64[ns]
    #Task 3 length                      int64
    #Task 4 length                      int64
    #dtype: object
    
    initial_date = '2019-10-01'

    #Create initialisation date 2019-10-01
    imported_data_df=imported_data_df.assign(Initialisation_Date = pd.to_datetime(initial_date))

    #create Task_1_length column (int)
    imported_data_df=imported_data_df.assign(Task_1_length = ((imported_data_df['Task 1 completion date'] - imported_data_df['Initialisation_Date'])/np.timedelta64(1,'D')).astype(int))

    #creating Task_2_lenght on the assumption that task two can start only after task 1 is completed. is that the case?
    imported_data_df=imported_data_df.assign(Task_2_length = ((imported_data_df['Task 2 Completion date'] - imported_data_df['Task 1 completion date'])/np.timedelta64(1,'D')).astype(int))
    #df['diff_days'] = df['End_date'] - df['Start_date']
    #df['diff_days']=df['diff_days']/np.timedelta64(1,'D')


    #find max(Task 2 Completion date) and define Transition_Date_Start as max(Task 2 Completion date) +5
    imported_data_df=imported_data_df.assign(Max_Preparation_Task_2_dt = max(imported_data_df['Task 2 Completion date']))
    #imported_data_df=imported_data_df.assign(Transition_Date_Start =  imported_data_df['Max_Preparation_Task_2_dt'] + 5 )
    imported_data_df["Transition_Date_Start"] = imported_data_df["Max_Preparation_Task_2_dt"] + timedelta(days=5)
    
    #compute waiting days between task2 end date and transition start date 
    wait_days = imported_data_df['Transition_Date_Start'] - imported_data_df['Task 2 Completion date']
    #convert the days into integer format
    Pause_length=[]
    for i in wait_days:
        j = i.days
        Pause_length.append(j)
    #Add this column to the dataframe 
    imported_data_df['Pause_length'] = Pause_length
    

    
    '''Define the data'''
    df = imported_data_df[['Organisation','Task_1_length', 'Task_2_length','Pause_length','Task 3 length','Task 4 length']]
    
    jobs_data = []
    for j in range (1, len(df.columns)):
        ls = []
        for i in range (0,len(df.index)):
            task = (i+1, df.iloc[i,j])
            ls.append(task)
        jobs_data.append(ls)
    
    org_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_organisations = range(org_count)
    
    
    '''Define Variables'''
    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')
    
    # Create job intervals and add to the corresponding machine lists.
    all_tasks = {}
    org_to_intervals = collections.defaultdict(list)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                organisation = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = model.NewIntVar(0, int(horizon), 'start' + suffix)
                end_var = model.NewIntVar(0, int(horizon), 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var)
                org_to_intervals[organisation].append(interval_var)
    
    
    '''Define Constraints'''
    # Create and add disjunctive constraints.
    for organisation in all_organisations:
        model.AddNoOverlap(org_to_intervals[organisation])

    
    '''Define the Objective'''
    # Makespan objective.
    obj_var = model.NewIntVar(0, int(horizon), 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)
    
    
    '''Declare the Solver'''
    # Solve model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    '''Display the results'''
    # Create one list of assigned tasks per machine.
    assigned_jobs = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            organisation = task[0]
            assigned_jobs[organisation].append(
                assigned_task_type(
                    start=solver.Value(all_tasks[job_id, task_id].start),
                    job=job_id,
                    index=task_id,
                    duration=task[1]))

    # Create per organisation output lines.
    output = ''
    for organisation in all_organisations:
        # Sort by starting time.
        assigned_jobs[organisation].sort()
        sol_line_tasks = 'Organisation ' + str(organisation) + ': '
        sol_line = '           '

        for assigned_task in assigned_jobs[organisation]:
            name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
            # Add spaces to output to align columns.
            sol_line_tasks += '%-10s' % name

            start = assigned_task.start
            duration = assigned_task.duration
            sol_tmp = '[%i,%i]' % (start, start + duration)
            # Add spaces to output to align columns.
            sol_line += '%-10s' % sol_tmp

        sol_line += '\n'
        sol_line_tasks += '\n'
        output += sol_line_tasks
        output += sol_line

    # Finally print the solution found.
    print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
    print(output)

     #Visualisation
    
    #Declaring variables which will store the solver data to be plotted later
    StartTimes = []
    Durations = []
    EndTimes = []
    OrganisationName = []
    job = []
    task = []
    loopnumber = []
    x=0
    Org = 0
    OrgNumber = []
    
    #loop over all organisationa and jobs/tasks to fill arrays with solver data
    for organisation in all_organisations:
            for assigned_task in assigned_jobs[organisation]:
                start = assigned_task.start
                duration = assigned_task.duration
                StartTimes.append(start)
                Durations.append(duration)
                OrganisationName.append(organisation)
                EndTimes.append(start+duration)
                job.append(assigned_task.job)
                task.append(assigned_task.index)
                OrgNumber.append(Org)
                loopnumber.append(x)
                x=x+1
            Org = Org + 1        
    
    #
    Distinctjob = np.unique(job)
    
    #Colour is based on a scale of 0-1, so I use their job number to define their colour and give each job a seperate colour
    No_jobs = max(job)
    Colour = np.divide(job,No_jobs)

    # Declaring a figure "gnt" 
    fig, gnt = plt.subplots() 

    # Setting Y-axis limits 
    gnt.set_ylim(0, 10*max(OrgNumber)+15) 

    # Setting X-axis limits 
    gnt.set_xlim(min(StartTimes)-10, max(EndTimes)+35) 

    # Setting labels for x-axis and y-axis 
    gnt.set_xlabel('Date') 
    gnt.set_ylabel('Organisation') 

    #Finding distinct organisations and numbering so that I can plot their name on the Y-axis
    DistinctOrgName = np.unique(OrganisationName)
    DistinctOrgNumber = np.unique(OrgNumber)
    YTicks = DistinctOrgNumber*10+5
    # Setting ticks on y-axis 
    gnt.set_yticks(YTicks)
    # Labelling tickes of y-axis 
    gnt.set_yticklabels(DistinctOrgName) 

    # Setting graph attribute 
    gnt.grid(True) 

    #Looping over all tasks numbers to create their bar on the chart
    import matplotlib.patches as mpatches
    for y in loopnumber:
            gnt.broken_barh([(StartTimes[y], Durations[y])], (10*OrgNumber[y], 9),facecolors =(Colour[y],0.1,1-Colour[y]))
            y=y+1
    
    #Looping over all distinct jobs to label them with their colour in the legend
    patchList = []
    for j in Distinctjob:
            data_key = mpatches.Patch(color=(j/No_jobs,0.1,1-j/No_jobs), label='Job ' + str(j))
            patchList.append(data_key)
    plt.legend(handles=patchList)



# In[21]:


MinimalJobshopToy()


# In[22]:


## TODO: 

# Convert the outputs from the solver into dates
# Get the visualisation working. (Build out two visulations one for high level and one for lower level tasks).
# General tidy up of the code (make more generalisable as we go forward)
# Organisation numbers not lined up exactly?

