from matplotlib.pylab import fftfreq
import matplotlib.pyplot as plt
import sqlite3
import datetime
import numpy as np
import statistics
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import plotly.express as px
import pandas as pd
import pandas as pd

#sensor profile
def plot_sensor_values(*exp_ids):
    """Define the function to plot sensor values over measurement time for multiple experiments"""
    for exp_id in exp_ids:
        # Connect to the SQLite database
        conn = sqlite3.connect('myexperiment.db')
        cur = conn.cursor()

        # Query the data, excluding the first two measurement_id for each experiment
        cur.execute("SELECT measurement_time, sensor_value FROM measurement_values WHERE exp_id = ? AND measurement_id > 5 ORDER BY measurement_time",(exp_id,))

        # Fetch the results
        results = cur.fetchall()

        # Close the connection
        conn.close()

        # Separate the results into two lists
        measurement_time = [row[0] for row in results]
        sensor_values = [row[1] for row in results]

        # Convert the measurement time to datetime objects
        measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_time]

        # Find duplicate measurement time and add 0.5s to the duplicate time
        ref_t = 0
        for i in range(len(measurement_time)):
            if measurement_time[i] == ref_t:
                measurement_time[i] += datetime.timedelta(seconds=0.5)
            ref_t = measurement_time[i]

        # Convert the modified measurement time to seconds from the start of the experiment
        start_time = measurement_time[0]
        measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

        # Plot the data using plotly express
        fig = px.line(x=measurement_time, y=sensor_values, title=f"Sensor Values Over Measurement Time for experiment {exp_id}",labels={"x": "Time [s]", "y": "Sensor Value"})
        fig.add_scatter(x=measurement_time, y=sensor_values, mode='markers', name='Measurement Points')
        fig.show()
        
def plot_sensor_values_for_phase(exp_id, machine_phase):
    """Define the function to plot sensor values over measurement time for a certain phase"""
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get sensor values and measurement time for a certain exp_id and phase
    query = f"SELECT sensor_value, measurement_time FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = '{machine_phase}'"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Unpack sensor values and measurement times
        sensor_values, measurement_times = zip(*results)

        # Convert the measurement times to datetime objects
        measurement_times = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_times]

        # Find duplicate measurement time and add 0.5s to the duplicate time
        ref_t = 0
        for i in range(len(measurement_times)):
            if measurement_times[i] == ref_t:
                measurement_times[i] += datetime.timedelta(seconds=0.5)
            ref_t = measurement_times[i]

        # Convert the measurement times to seconds from the start of the experiment
        start_time_experiment = min(measurement_times)
        measurement_times = [(t - start_time_experiment).total_seconds() for t in measurement_times]

        # Plot sensor values over measurement time using plotly express
        #fig = px.line(x=measurement_times, y=sensor_values, labels={'x':'Time [s]', 'y':'Sensor Value'}, title=f'Sensor Values for exp_id {exp_id} during {machine_phase} machine_phase')
        #fig.add_scatter(x=measurement_times, y=sensor_values, mode='markers', name='Measurement Points')
        #fig.show()

        # Plot the data using matplotlib
        plt.plot(measurement_times, sensor_values)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Sensor Value', fontsize=12)
        plt.xticks(np.arange(0, 60, 20), fontsize=12)
        plt.yticks(np.arange(0, 200000, 10000), fontsize=12)
        plt.title(f"Sensor Value vs Time for Mass = 1059", fontsize=16)
        plt.rcParams['font.family'] = 'Arial'
        plt.show()
    else:
        print(f"No data found for exp_id {exp_id} during {machine_phase} phase")

def plot_sensor_values_for_specific_time(exp_id: str, start_time_seconds: int, end_time_seconds: int):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get sensor values for a certain exp_id and phase
    query = f"SELECT sensor_value, measurement_time FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = 'no_fabric'"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Unpack sensor values and measurement times
        sensor_values, measurement_time = zip(*results)

        # Convert measurement time to datetime objects
        measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in measurement_time]

        # Find duplicate measurement time and add 0.5s to the duplicate time
        ref_t = 0
        for i in range(len(measurement_time)):
            if measurement_time[i] == ref_t:
                measurement_time[i] += datetime.timedelta(seconds=0.5)
            ref_t = measurement_time[i]

        # Convert measurement time to seconds from the start of the experiment
        start_time = measurement_time[0]
        measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

        # Find the indices of the sensor values within the specified time range
        indices = [i for i, t in enumerate(measurement_time) if start_time_seconds <= t <= end_time_seconds]

        # Extract sensor values within the specified time range
        sensor_values = [sensor_values[i] for i in indices]
        measurement_time = [measurement_time[i] for i in indices]

        # Plot the sensor values over the specified time range
        plt.plot(measurement_time, sensor_values)
        plt.xlabel('Time [s]')
        plt.ylabel('Sensor Value [-]')
        plt.title('Sensor Values Over Specified Time Range')
        plt.show()

#sensor calibration
def sensor_to_force_value(exp_id: str, machine_phase: str, weight: float):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get sensor values for a certain exp_id and phase
    query = f"SELECT sensor_value FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = '{machine_phase}'"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # If results are not empty
    if results:
        # Unpack sensor values
        sensor_values = np.array([value[0] for value in results])

        # Calculate mean and standard deviation using numpy
        mean_value = np.mean(sensor_values)
        std_dev = np.std(sensor_values)

        # Calculate percentage error
        CV = std_dev / mean_value * 100

        # Print the mean, standard deviation and percentage error
        print(f"Mean: {mean_value}, Standard Deviation: {std_dev}, Coefficient of Variation: {CV}%")

        # Insert the values into the sensor_calibration table
        insert_query = f"INSERT INTO sensor_cal (exp_id, mean_, std_dev, CV, weight_gram) VALUES ('{exp_id}', {mean_value}, {std_dev}, {CV}, {weight})"
        cursor.execute(insert_query)
        conn.commit()
        conn.close()

        return mean_value, std_dev, CV

    else:
        print(f"No data found for exp_id {exp_id} during {machine_phase} phase")
        return None, None, None

def plot_force_against_mean():
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get mean_, weight, and force from sensor_calibration table
    query = "SELECT mean_, weight_gram, force_newton, exp_id FROM sensor_cal"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Unpack mean_, weight, force, and exp_id
        mean_values, weights, forces, exp_ids = zip(*results[:15])

        # Convert to numpy arrays for plotting
        forces_np = np.array(forces)
        mean_values_np = np.array(mean_values)

        # Create a linear regression model for force
        force_model = LinearRegression()
        force_model.fit(mean_values_np.reshape(-1, 1), forces_np)

        # Make predictions using the model
        force_predictions = force_model.predict(mean_values_np.reshape(-1, 1))

        # Plot the data and the regression line
        plt.scatter(mean_values, forces, color='blue', label='Force')
        plt.plot(mean_values, force_predictions, color='red', label='Force Regression')
        plt.xlabel('Mean Sensor Value')
        plt.ylabel('Force [N]')
        plt.title('Force Against Mean Sensor Value')
        plt.legend()
        for i in range(len(mean_values)):
            plt.text(mean_values[i], forces[i], exp_ids[i], fontsize=9)

        plt.show()

        # Print the line equation and the coefficient of determination (R^2)
        print(f"Line equation: y = {force_model.coef_[0]}x + {force_model.intercept_}")
        print(f"Coefficient of determination (R^2): {force_model.score(mean_values_np.reshape(-1, 1), forces_np)}")

        # Calculate the standard error of the linear regression line
        residuals = forces_np - force_predictions
        squared_residuals = residuals ** 2
        mean_squared_error = np.mean(squared_residuals)
        standard_error = np.sqrt(mean_squared_error)

        print(f"Standard Error of the Linear Regression Line: {standard_error}")
    else:
        print("No data found in sensor_cal table")

def plot_callibration(exp_id, machine_phase, weight):
    """Define the function to plot sensor values over measurement time for a certain phase"""
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get sensor values and measurement time for a certain exp_id and phase
    query = f"SELECT sensor_value, measurement_time FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = '{machine_phase}'"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Unpack sensor values and measurement times
        sensor_values, measurement_times = zip(*results)

        # Convert the measurement times to datetime objects
        measurement_times = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_times]

        # Find duplicate measurement time and add 0.5s to the duplicate time
        ref_t = 0
        for i in range(len(measurement_times)):
            if measurement_times[i] == ref_t:
                measurement_times[i] += datetime.timedelta(seconds=0.5)
            ref_t = measurement_times[i]

        # Convert the measurement times to seconds from the start of the experiment
        start_time_experiment = min(measurement_times)
        measurement_times = [(t - start_time_experiment).total_seconds() for t in measurement_times]


        # Plot sensor values over measurement time using plotly express
        #fig = px.line(x=measurement_times, y=sensor_values, labels={'x':'Time [s]', 'y':'Sensor Value'}, title=f'Sensor Values for exp_id {exp_id} during {machine_phase} machine_phase')
        #fig.add_scatter(x=measurement_times, y=sensor_values, mode='markers', name='Measurement Points')
        #fig.show()
        
        # Calculate the standard deviation of sensor values
        std_deviation = np.std(sensor_values)

        # Plot the sensor values with shaded area representing the standard deviation
        plt.plot(measurement_times, sensor_values)
        plt.fill_between(measurement_times, sensor_values - std_deviation, sensor_values + std_deviation, color='grey', alpha=0.5)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Sensor Value [-]', fontsize=12)
        plt.xticks(np.arange(0, 35, 5), fontsize=12)
        plt.xlim(0, 30)  # Set the x-axis limits from 0 to 30 seconds
        plt.yscale('log')
        plt.yticks([1000, 10000, 100000, 200000], ["1k", "10k", "100k", "200k"], fontsize=12)

        plt.title(f"Sensor Value vs Time for Mass = {weight}g", fontsize=16)
        plt.rcParams['font.family'] = 'Arial'
        plt.tight_layout()
        plt.savefig('sensor_values_plot.png', dpi=200)
        plt.show()
        
    else:
        print(f"No data found for exp_id {exp_id} during {machine_phase} phase")

#force profile
def plot_force_values(*exp_ids):
    """Define the function to plot sensor values over measurement time for multiple experiments"""
    for exp_id in exp_ids:
        # Connect to the SQLite database
        conn = sqlite3.connect('myexperiment.db')
        cur = conn.cursor()

        # Query the data, excluding the first two measurement_id for each experiment
        cur.execute("SELECT measurement_time, sensor_value FROM measurement_values WHERE exp_id = ? AND measurement_id > 5 ORDER BY measurement_time",(exp_id,))

        # Fetch the results
        results = cur.fetchall()

        # Close the connection
        conn.close()

        # Separate the results into two lists
        measurement_time = [row[0] for row in results]
        sensor_values = [row[1] for row in results]

        # Convert the measurement time to datetime objects
        measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_time]

        # Find duplicate measurement time and add 0.5s to the duplicate time
        ref_t = 0
        for i in range(len(measurement_time)):
            if measurement_time[i] == ref_t:
                measurement_time[i] += datetime.timedelta(seconds=0.5)
            ref_t = measurement_time[i]

        # Convert the modified measurement time to seconds from the start of the experiment
        start_time = measurement_time[0]
        measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

        # Convert sensor values to force values
        force_values = [(0.0002* x ) for x in sensor_values]
        # Plot the data using matplotlib
        plt.plot(measurement_time, force_values)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Force [N]', fontsize=12)
        plt.title(f"Force vs Time for FL=1, RPM=40", fontsize=16)
        plt.rcParams['font.family'] = 'Arial'
        plt.tight_layout()
        plt.show()       

def plot_peak_force_values(*exp_ids):
    """Define the function to plot sensor values over measurement time for multiple experiments"""
    for exp_id in exp_ids:
        # Connect to the SQLite database
        conn = sqlite3.connect('myexperiment.db')
        cur = conn.cursor()

        # Query the data, excluding the first two measurement_id for each experiment
        cur.execute("SELECT measurement_time, sensor_value FROM measurement_values WHERE exp_id = ? AND measurement_id > 5 AND sensor_value ORDER BY measurement_time",(exp_id,))

        # Fetch the results
        results = cur.fetchall()

        # Close the connection
        conn.close()

        # Separate the results into two lists
        measurement_time = [row[0] for row in results]
        sensor_values = [row[1] for row in results]

        # Convert the measurement time to datetime objects
        measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_time]

        # Find duplicate measurement time and add 0.5s to the duplicate time
        ref_t = 0
        for i in range(len(measurement_time)):
            if measurement_time[i] == ref_t:
                measurement_time[i] += datetime.timedelta(seconds=0.5)
            ref_t = measurement_time[i]

        # Convert the modified measurement time to seconds from the start of the experiment
        start_time = measurement_time[0]
        measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

        # Convert sensor values to force values
        force_values = [(0.0002* int(x) ) for x in sensor_values]
        
        # Find peaks in the force values
        peaks, _ = find_peaks(force_values)

        # Extract peak force values
        peak_force_values = [force_values[i] for i in peaks if force_values[i] > 10]

        # Plot the data using matplotlib
        plt.plot(measurement_time, force_values)
        plt.scatter([measurement_time[i] for i in peaks if force_values[i] > 10], peak_force_values, color='red', label='Peaks')
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Force [N]', fontsize=12)
        plt.yticks(np.arange(0, 70, 10), fontsize=12)
        plt.title(f"Force vs Time for FL=1, RPM=80", fontsize=16)
        plt.rcParams['font.family'] = 'Arial'
        plt.tight_layout()
        
        # Label the peaks
        x_label = [measurement_time[i] for i in peaks if force_values[i] > 10]
        for i,v in enumerate(peak_force_values):
            plt.annotate(str(round(x_label[i],2)),xy=(x_label[i],v))

        
        plt.show()
        
#analysis
def calculate_peak_std_deviation(exp_id: str, machine_phase: str):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get sensor values and measurement time for a certain exp_id and phase
    query = f"SELECT sensor_value, measurement_time FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = '{machine_phase}'"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Unpack sensor values and measurement times
        sensor_values, measurement_time = zip(*results)

        # Convert measurement time to datetime objects
        measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_time]

        # Convert measurement time to seconds from the start of the experiment
        start_time = measurement_time[0]
        measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

        # Convert sensor values to force values
        force_values = [(0.0002 * x ) for x in sensor_values]

        # Find peaks in the force values
        peaks, _ = find_peaks(force_values)

        # Extract peak force values
        peak_force_values = [force_values[i] for i in peaks]

        # Filter peak force values greater than 10 N
        filtered_peak_force_values = [value for value in peak_force_values if value > 10]

        # Plot peak force values against time
        plt.plot(range(1, len(filtered_peak_force_values) + 1), filtered_peak_force_values)
        plt.xlabel('Time [s]')
        plt.ylabel('Peak Force [N]')
        plt.title('Peak Force Values Against Time')
        #plt.show()

        # Calculate peak mean
        peak_mean = np.mean(filtered_peak_force_values)
        # Print the peak standard deviation
        print(f"Peak Mean: {peak_mean}")

        # Calculate peak standard deviation
        peak_std_deviation = np.std(filtered_peak_force_values)

        # Print the peak standard deviation
        print(f"Peak Standard Deviation: {peak_std_deviation}")

        # Calculate percentage error
        percentage_error = (peak_std_deviation / peak_mean) * 100

        # Print the percentage error
        print(f"Percentage Error: {percentage_error}%")

        return peak_mean, peak_std_deviation, percentage_error
    else:
        print(f"No data found for exp_id {exp_id} during {machine_phase} phase")
        return None, None, None

def calculate_peak_trough_mean(exp_id: str, machine_phase: str):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get sensor values and measurement time for a certain exp_id and phase
    query = f"SELECT sensor_value, measurement_time FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = '{machine_phase}'"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Unpack sensor values and measurement times
        sensor_values, measurement_time = zip(*results)

        # Convert measurement time to datetime objects
        measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_time]

        # Find duplicate measurement time and add 0.5s to the duplicate time
        ref_t = 0
        for i in range(len(measurement_time)):
            if measurement_time[i] == ref_t:
                measurement_time[i] += datetime.timedelta(seconds=0.5)
            ref_t = measurement_time[i]

        # Convert measurement time to seconds from the start of the experiment
        start_time = measurement_time[0]
        measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

        # Convert sensor values to force values
        force_values = [(0.0002 * x ) for x in sensor_values]

        # Identify peaks in the force values
        peaks, _ = find_peaks(force_values, distance= 50)

        # Extract peak values
        peak_values = [force_values[i] for i in peaks]
        

        # Calculate the mean of the peak values above 10
        peak_values_above_10 = [peak for peak in peak_values if peak > 10]
        peak_mean = statistics.mean(peak_values_above_10)
        peak_std_deviation = statistics.stdev(peak_values_above_10)


        # Identify troughs in the force values
        inverted_force_values = [-x for x in force_values]
        troughs, _ = find_peaks(inverted_force_values, distance=100)

        # Extract trough values
        trough_values = [force_values[i] for i in troughs]

        # Calculate the mean of the trough values
        trough_mean = statistics.mean(trough_values)

        

        # Plot the force values over measurement time
        plt.plot(measurement_time, force_values)
        plt.xlabel('Measurement Time (s)')
        plt.ylabel('Force Value')
        plt.title('Force Values Over Measurement Time for Experiment 7')

        # Plot the mean of the peak values
        plt.axhline(y=peak_mean, color='r', linestyle='--')
        #plt.text(max(measurement_time), peak_mean, f"Mean: {peak_mean}", color='r', ha='right', va='baseline')
       
        # Plot the peaks
        plt.plot([measurement_time[i] for i in peaks if force_values[i] > 10], [force_values[i] for i in peaks if force_values[i] > 10], 'ro', label='Peaks')
        #plt.axhline(y=trough_mean, color='b', linestyle='--')
        plt.show()

        print(f"The mean of the peak values is: {peak_mean}")
        print(f"The mean of the trough values is: {trough_mean}")
        print(f"The standard deviation of the peak values is: {peak_std_deviation}")

        return peak_mean, trough_mean

    else:
        print(f"No data found for exp_id {exp_id} during {machine_phase} phase")
        return None

def plot_lin_reg(*exps):
    num_i = 0
    exp_names = ['FL1', 'FL2', 'FL3']
    my_legends = []
    for exp_ids in exps:
        machine_RPM = []
        peak_means = []
        for exp_id in exp_ids:
            # Connect to the SQLite database
            conn = sqlite3.connect('myexperiment.db')
            cursor = conn.cursor()

            # Query to get sensor values and measurement time for a certain exp_id and phase
            query = f"SELECT sensor_value, measurement_time FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = 'with_fabric'"
            cursor.execute(query)

            # Fetch all rows from the last executed query
            results = cursor.fetchall()

            # If results are not empty
            if results:
                # Unpack sensor values and measurement times
                sensor_values, measurement_time = zip(*results)

                # Convert measurement time to datetime objects
                measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_time]

                # Find duplicate measurement time and add 0.5s to the duplicate time
                ref_t = 0
                for i in range(len(measurement_time)):
                    if measurement_time[i] == ref_t:
                        measurement_time[i] += datetime.timedelta(seconds=0.5)
                    ref_t = measurement_time[i]

                # Convert measurement time to seconds from the start of the experiment
                start_time = measurement_time[0]
                measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

                # Convert sensor values to force values
                force_values = [(0.0002 * x) for x in sensor_values]

                # Identify peaks in the force values above 10 N
                peaks, _ = find_peaks(force_values, height=10)

                # Extract peak values
                peak_values = [force_values[i] for i in peaks]

                # Calculate the mean of the peak values
                peak_mean = statistics.mean(peak_values)

                # Query to get material_id for a certain exp_id from the experiment table
                query = f"SELECT machineparameter_id FROM experiment WHERE exp_id = '{exp_id}'"
                cursor.execute(query)
                machineparameter_id = cursor.fetchone()[0]

                # Query to get RPM for the fetched machineparameter_id from the machine_parameter table
                query = f"SELECT RPM FROM machine_parameter WHERE machineparameter_id = '{machineparameter_id}'"
                cursor.execute(query)

                # Fetch the RPM from the last executed query
                RPM = cursor.fetchone()[0]

                # Append the peak mean and RPM to the lists
                peak_means.append(peak_mean)
                machine_RPM.append(RPM)

            # Close the connection
            conn.close()

        # Create a scatter plot of peak means against machine RPM
        plt.scatter(machine_RPM, peak_means, marker='o')

        # Perform linear regression
        model = LinearRegression()
        X = np.array(machine_RPM).reshape(-1, 1)
        y = np.array(peak_means)
        model.fit(X, y)

        # Get the regression coefficients
        slope = model.coef_[0]
        intercept = model.intercept_

        # Get the R-squared value
        r_squared = model.score(X, y)

        # Get the standard error of the estimate
        y_pred = model.predict(X)
        std_error = np.sqrt(mean_squared_error(y, y_pred))

        # Print the regression results
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"R-squared: {r_squared}")
        print(f"Standard Error: {std_error}")

        def myfunc(x):
            return slope * x + intercept

        mymodel = list(map(myfunc, X))
        plt.plot(X, mymodel)

        my_legends.append(f"Peak means: {exp_names[num_i]}")
        my_legends.append(f"LRegression: {exp_names[num_i]}")
        num_i += 1

    print(my_legends)
    plt.legend(my_legends)
    plt.xlabel('Machine RPM')
    plt.ylabel('Force [N]')
    plt.title('Peak Mean Force vs Machine RPM')
    plt.xticks([20, 40, 80])
    plt.yticks(np.arange(0, 500, 50))

    plt.show()

def calculate_frequency_domain(exp_id: str, machine_phase: str):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get sensor values for a certain exp_id and phase
    query = f"SELECT sensor_value FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = '{machine_phase}'"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Unpack sensor values
        sensor_values = [result[0] for result in results]

        # Apply FFT to the sensor values
        yf = fft(sensor_values)

        # Calculate the absolute value of each complex number in the result
        yf_abs = np.abs(yf)

        # Calculate the frequencies for the FFT
        xf = fftfreq(len(sensor_values))

        # Plot the absolute values of the FFT results against the frequencies
        plt.plot(xf, yf_abs)
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title('Frequency Domain')
        plt.show()
    else:
        print(f"No data found for exp_id {exp_id} during {machine_phase} phase")

def plot_peaks(exp_id: str, machine_phase: str = None):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get sensor values and measurement time for a certain exp_id and phase
    if machine_phase:
        query = f"SELECT sensor_value, measurement_time FROM measurement_values WHERE exp_id = '{exp_id}' AND machine_phase = '{machine_phase}'"
    else:
        query = f"SELECT sensor_value, measurement_time FROM measurement_values WHERE exp_id = '{exp_id}' AND measurement_id > 5"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Unpack sensor values and measurement times
        sensor_values, measurement_time = zip(*results)

        # Convert measurement time to datetime objects
        measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_time]

        # Convert measurement time to seconds from the start of the experiment
        start_time = measurement_time[0]
        measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

        # Convert sensor values to force values
        force_values = [(0.0002 * x ) for x in sensor_values]

        # Find peaks in the force values
        peaks, _ = find_peaks(force_values)

        # Extract peak force values
        peak_force_values = [force_values[i] for i in peaks]

        # Filter peak force values greater than 10 N
        filtered_peak_force_values = [value for value in peak_force_values if value > 10]

        return filtered_peak_force_values

def plot_peak_means_against_layer(*exp_ids):
    # Create an empty list to store the peak force values for each experiment
    all_peaks = []

    # Iterate over each exp_id
    for exp_id in exp_ids:
        # Call the plot_peaks function to get the peak force values for the current exp_id
        peaks = plot_peaks(exp_id, 'with_fabric')

        # Append the peak force values to the all_peaks list
        all_peaks.append(peaks)

    # Find the minimum number of points among all experiments
    min_num_points = min(len(peaks) for peaks in all_peaks)

    # Cut the remaining points for each experiment
    all_peaks = [peaks[:min_num_points] for peaks in all_peaks]

    # Calculate the mean and standard deviation for each experiment
    means = [sum(peaks) / min_num_points for peaks in all_peaks]
    std_devs = [np.std(peaks) for peaks in all_peaks]

    # Generate x values for the plot
    x_values = range(1, min_num_points + 1)

    # Plot the peak force values against time for each experiment
    for i, peaks in enumerate(all_peaks):
        plt.plot(x_values, peaks, label=f'Experiment {i+1}')
        plt.fill_between(x_values, means[i] - std_devs[i], means[i] + std_devs[i], alpha=0.3, label=f'Std Dev {i+1}: {std_devs[i]:.2f}')
        plt.axhline(means[i], linestyle='--', label=f'Mean {i+1}: {means[i]:.2f}')

    plt.xlabel('Datapoint')
    plt.ylabel('Peak Force [N]')
    plt.title('Peak Force Values')
    plt.legend()
    plt.show()

def plot_boxplot(exp_ids):
    # Create an empty list to store the peak force values for each experiment
    all_peaks = []

    # Iterate over each exp_id
    for exp_id in exp_ids:
        # Call the plot_peaks function to get the peak force values for the current exp_id
        peaks = plot_peaks(exp_id, 'with_fabric')

        # Append the peak force values to the all_peaks list
        all_peaks.append(peaks)

    # Create a box plot
    plt.boxplot(all_peaks)

    # Set the labels and title
    plt.xlabel('Experiment ID')
    plt.ylabel('Force [N]')
    plt.title('Peak Force Distribution')

    # Show the plot
    plt.show() 

def plot_tensile_strength(exp_ids):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Create an empty list to store the tensile strength values for each experiment
    all_tensile_strengths = []

    # Iterate over each exp_id
    for exp_id in exp_ids:
        # Query to get the tensile strength for the current exp_id
        query = f"SELECT Fmax1, Fmax2, Fmax3, Fmax4, Fmax5 FROM tensile_strength WHERE exp_id = '{exp_id}'"
        cursor.execute(query)

        # Fetch the tensile strength values from the last executed query
        result = cursor.fetchone()

        # If result is not empty
        if result:
            # Append the tensile strength values to the all_tensile_strengths list
            all_tensile_strengths.append(result)

            # Append the tensile strength value to the all_tensile_strengths list
            #ll_tensile_strengths.append(tensile_strength)

    # Close the connection
    conn.close()

    # Create a box plot
    plt.boxplot(all_tensile_strengths)

    # Set the labels and title
    plt.xlabel('Experiment ID')
    plt.ylabel('Tensile Strength')
    plt.title('Tensile Strength Distribution')

    # Show the plot
    plt.show()

def clean(*exp_ids):
    """Define the function to plot sensor values over measurement time for multiple experiments"""
    for exp_id in exp_ids:
        # Connect to the SQLite database
        conn = sqlite3.connect('myexperiment.db')
        cur = conn.cursor()

        # Query the data, excluding the first two measurement_id for each experiment
        cur.execute("SELECT measurement_time, sensor_value FROM measurement_values WHERE exp_id = ? AND measurement_id > 5 ORDER BY measurement_time",(exp_id,))

        # Fetch the results
        results = cur.fetchall()

        # Close the connection
        conn.close()

        # Separate the results into two lists
        measurement_time = [row[0] for row in results]
        sensor_values = [row[1] for row in results]

        # Convert the measurement time to datetime objects
        measurement_time = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f") for t in measurement_time]

        # Find duplicate measurement time and add 0.5s to the duplicate time
        ref_t = 0
        for i in range(len(measurement_time)):
            if measurement_time[i] == ref_t:
                measurement_time[i] += datetime.timedelta(seconds=0.5)
            ref_t = measurement_time[i]

        # Convert the modified measurement time to seconds from the start of the experiment
        start_time = measurement_time[0]
        measurement_time = [(t - start_time).total_seconds() for t in measurement_time]

        # Convert sensor values to force values
        force_values = [(0.0002* x ) for x in sensor_values]
        

        # Plot the data using matplotlib
        start_point,end_point = 100,480
        #plt.plot([t for t in measurement_time if 50 < t < 300], [f for t, f in zip(measurement_time, force_values) if 50 < t < 300])
        start_index = [i for i,x in enumerate(measurement_time) if x>start_point][0]
        end_index = [i for i,x in enumerate(measurement_time) if x>end_point][0] 

        plt.plot(measurement_time[start_index:end_index], force_values[start_index:end_index])
        '''''
        # Find peaks in the force values
        peaks, _ = find_peaks(force_values)

        # Extract peak force values
        min_force,max_force = 200,350
        peak_force_values = [force_values[i] for i in peaks if (force_values[i] > min_force) and (force_values[i] < max_force)]
        peak_time_values = [measurement_time[i] for i in peaks if (force_values[i] > min_force) and (force_values[i] < max_force)]

        start_index = [i for i,x in enumerate(peak_time_values) if x>start_point][0]
        end_index = [i for i,x in enumerate(peak_time_values) if x>end_point][0] 

        #store_peaks_sql(peak_force_values[start_index:end_index],exp_id)
        plt.scatter(peak_time_values[start_index:end_index], peak_force_values[start_index:end_index], color='red', label='Peaks')

        # Plot the peaks as dots
        '''
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Force [N]', fontsize=12)
        plt.title(f"Force vs Time ", fontsize=16)
        plt.rcParams['font.family'] = 'Arial'
        plt.tight_layout()
        plt.show()

#mahlo
def calculate_mean_module_values(batch_number):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Query to get the value and module for the specified batch number
    query = f"SELECT value, module_id FROM value WHERE batchNumber = '{batch_number}'"
    cursor.execute(query)

    # Fetch all rows from the last executed query
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    # If results are not empty
    if results:
        # Create dictionaries to store the sum and count of values for each module
        sum_values = {}
        count_values = {}

        # Iterate over the results
        for value, module in results:
            # If the module is not in the dictionaries, initialize it
            if module not in sum_values:
                sum_values[module] = 0
                count_values[module] = 0

            # Add the value to the sum and increment the count for the module
            sum_values[module] += value
            count_values[module] += 1

        # Create lists to store the mean values and modules
        mean_values = []
        mean_modules = []

        # Iterate over the modules
        for module in range(10,400):
            # If the module has values, calculate the mean and append to the lists
            if module in sum_values:
                mean_value = (sum_values[module]*1000) / count_values[module]
                mean_values.append(mean_value)
                mean_modules.append(module)

        # Plot mean_value against mean_module
        plt.plot(mean_modules, mean_values)
        plt.xlabel('Module')
        plt.ylabel('Fabric Weight [g/$m^2$]')

        plt.title(f'Fabric Weight [g/$m^2$] ')
        plt.show()

    else:
        print(f"No data found for batch number {batch_number}")

def plot_peak_mean_trendline(exp_id):
    # Call the plot_peaks function to get the peak force values for the current exp_id
    peaks = plot_peaks(exp_id,'with_fabric')

    # Generate x values for the plot
    x_values = range(1, len(peaks) + 1)

    # Calculate the trendline using a moving average
    window_size = 25
    trendline = np.convolve(peaks, np.ones(window_size), 'valid') / window_size

    # Plot the peak force values with trendline
    plt.plot(x_values, peaks, color='lightgrey', label='Peak Force')
    plt.plot(x_values[window_size-1:], trendline, label='Moving Average')
    plt.xlabel('Datapoint')
    plt.ylabel('Force [N]')
    plt.title(f'Peak Needle Punching Force ')
    plt.legend()
    plt.show()

def store_peaks_sql(list_of_data, exp_id):
    cols = ["peak_val"]
    df = pd.DataFrame(list_of_data, columns=cols)
    df["exp_id"]=exp_id
    print(df)
    
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute("CREATE TABLE IF NOT EXISTS peaks (peak_val REAL, exp_id TEXT)")

    # Insert the data into the table
    for index, row in df.iterrows():
        cursor.execute("INSERT INTO peaks (peak_val, exp_id) VALUES (?, ?)", (row['peak_val'], row['exp_id']))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def calculate_mean_std_dev_clean():
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Specify the experiment IDs for which you want to calculate the mean and standard deviation
    experiment_ids = ['210','208','218','219','222','227','228','205']

    # Iterate over each experiment ID
    for experiment_id in experiment_ids:
        # Query to get the peak values for the current experiment ID
        cursor.execute("SELECT peak_val FROM peaks WHERE exp_id = ?", (experiment_id,))
        peak_values = cursor.fetchall()

        # Calculate the mean and standard deviation for the peak values
        mean = np.mean(peak_values)
        std_dev = np.std(peak_values)

        # Print the mean and standard deviation
        print(f"Experiment {experiment_id}: Mean = {mean}, Standard Deviation = {std_dev}")

def plot_boxplot_clean(exp_ids):
    # Connect to the SQLite database
    conn = sqlite3.connect('myexperiment.db')
    cursor = conn.cursor()

    # Create an empty list to store the peak force values for each experiment
    all_peaks = []

    # Iterate over each experiment ID
    for exp_id in exp_ids:
        # Query the peak values for the current experiment ID
        cursor.execute("SELECT peak_val FROM peaks WHERE exp_id = ?", (exp_id,))
        peak_values = cursor.fetchall()

        # Extract the peak force values from the fetched results
        peaks = [row[0] for row in peak_values]

        # Append the peak force values to the list
        all_peaks.append(peaks)

    # Close the connection
    conn.close()

    # Create a box plot
    plt.boxplot(all_peaks)

    # Set the labels and title
    plt.xlabel('Experiment')
    plt.ylabel('Force [N]')
    plt.title('Peak Force Distribution')

    # Show the plot
    plt.show()




if __name__ == "__main__":


    plot_sensor_values('210')

    #plot_sensor_values_for_phase('210', 'with_fabric')

    #plot_sensor_values_for_specific_time('210', 0, 100)

    #sensor_to_force_value('269', 'not_running', 0 )

    #plot_force_against_mean()

    #plot_callibration('260', 'with_fabric', 2097)

    #plot_force_values('209','with_fabric')

    #plot_peak_force_values('210')

    #calculate_peak_std_deviation('209', 'with_fabric')

    #calculate_peak_trough_mean('222', 'with_fabric')

    #plot_lin_reg(['204','209','210','205','212','211','207','213','214'],['208','218','219','206','215','216','217','220','221'],['222','227','228','223','225','226','224','229','230'])

    #calculate_frequency_domain('205', 'with_fabric')

    #plot_sine_graph('224', 'with_fabric')

    #plot_peaks('210', 'with_fabric')

    #plot_peak_means_against_layer('210')

    #plot_boxplot(exp_ids)

    #plot_tensile_strength(exp_ids)

    #calculate_mean_module_values('9')

    #plot_peak_mean_trendline('211')

    #clean('222')

    #calculate_mean_std_dev_clean()

    #plot_boxplot_clean(27)

    #plot_boxplot_clean(exp_ids)

    #same FL, different RPM:
    #['204','209','210','205','212','211','207','213','214'],['208','218','219','206','215','216','217','220','221'],['222','227','228','223','225','226','224','229','230']

    #same RPM, different FL:
    #['204','209','210','208','218','219','222','227','228'],['205','212','211','206','215','216','223','225','226'],['207','213','214','217','220','221','224','229','230']
        
    