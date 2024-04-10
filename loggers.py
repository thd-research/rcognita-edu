"""
Contains the logger interface along with concrete realizations for each separate system.

"""

from tabulate import tabulate

import csv

class Logger:
    """
    Interface class for data loggers.
    Concrete loggers, associated with concrete system-controller setups, are should be built upon this class.
    To design a concrete logger: inherit this class, override:
        | :func:`~loggers.Logger.print_sim_step` :
        | print a row of data of a single simulation step, typically into the console (required).
        | :func:`~loggers.Logger.log_data_row` :
        | same as above, but write to a file (required).
    
    """
    
    def print_sim_step():
        pass
    
    def log_data_row():
        pass
    
class Logger3WRobotNI(Logger):
    """
    Data logger for a 3-wheel robot with static actuators.
    
    """
    def print_sim_step(self, t, xCoord, yCoord, alpha, run_obj, accum_obj, action):
    # alphaDeg = alpha/np.pi*180      
    
        row_header = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'run_obj', 'accum_obj', 'v [m/s]', 'omega [rad/s]']  
        row_data = [t, xCoord, yCoord, alpha, run_obj, accum_obj, action[0], action[1]]  
        row_format = ('8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.1f', '8.3f', '8.3f')   
        table = tabulate([row_header, row_data], floatfmt=row_format, headers='firstrow', tablefmt='grid')
    
        print(table)
    
    def log_data_row(self, datafile, t, xCoord, yCoord, alpha, run_obj, accum_obj, action):
        with open(datafile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, xCoord, yCoord, alpha, run_obj, accum_obj, action[0], action[1]])
                