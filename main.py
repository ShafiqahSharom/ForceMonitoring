import serial
import sqlite3
import time
from datetime import datetime
import keyboard  

def flush_serial_buffer(ser):
    # Read and discard any available data in the serial buffer
    while ser.in_waiting > 0:
        ser.readline()
        
def insert_material_properties(conn, cur):
    material_type = input("Enter material_type: ")
    layer = int(input("Enter layer: "))
    QMP_id = int(input("Enter QMP_id: "))
    
     

    cur.execute("INSERT INTO material_properties (material_type, layer, QMP_id) VALUES (?, ?, ?)",
                (material_type, layer, QMP_id))
    conn.commit()

    return cur.lastrowid

def insert_machine_properties(conn, cur):
    # Get user input on machine parameter
    RPM = int(input("Enter machine RPM: "))
    needle_depth = int(input("Enter needle depth: "))
    needle_density = input("Enter needle density: ")

    # Insert user input into the machine_parameter table
    cur.execute("INSERT INTO machine_parameter (RPM, needle_depth, needle_density) VALUES (?, ?, ?)",
                (RPM, needle_depth, needle_density))
    
    conn.commit()

    return cur.lastrowid

def start_new_experiment(conn, cur, material_properties_id, machineparameter_id):

    # Create a new experiment entry
    exp_name = input("Enter experiment name: ")
    cur.execute("INSERT INTO experiment (exp_name, material_id, machineparameter_id) VALUES (?, ?, ?)",
                (exp_name, material_properties_id, machineparameter_id))
    conn.commit()

    return cur.lastrowid, material_properties_id, machineparameter_id

# Connect to SQLite database
conn = sqlite3.connect('myexperiment.db')
cur = conn.cursor()

while True:
    # Ask whether to insert new material properties or reuse existing ones
    insert_material = input("Do you want to insert new material properties? (yes/no): ").lower()

    if insert_material == 'yes':
        material_properties_id = insert_material_properties(conn, cur)
    else:
        material_id = int(input("Enter material_id: "))
        # Assuming material_id corresponds to an existing material in the material_properties table
        material_properties_id = material_id

    # Ask whether to insert new machine properties or reuse existing ones
    insert_machine = input("Do you want to insert new machine properties? (yes/no): ").lower()

    if insert_machine == 'yes':
        machineparameter_id = insert_machine_properties(conn, cur)
    else:
        machine_id = int(input("Enter machineparameter_id: "))
        # Assuming machine_id corresponds to an existing machine parameter in the machine_parameter table
        machineparameter_id = machine_id

    exp_id, _, _ = start_new_experiment(conn, cur, material_properties_id, machineparameter_id)
    ser = serial.Serial('COM3', 115200)
    measurement_id = 1

    # Ask the user to input the initial machine phase using numeric values
    init_machine_phase = input("Enter machine phase (1: not_running, 2: without_fabric, 3: with_fabric): ")
    if init_machine_phase  == '1':
        machine_phase = 'not_running'
    elif init_machine_phase  == '2':
        machine_phase = 'without_fabric'
    elif init_machine_phase  == '3':
        machine_phase = 'with_fabric'
    else:
        print("Invalid input. Setting machine phase to not_running.")
        machine_phase = 'not_running'

    # Flush out existing values in the serial buffer
    flush_serial_buffer(ser)

    while True:
        if ser.in_waiting > 0:
            sensor_value = ser.readline().decode('utf-8').rstrip()
            parts = sensor_value.split(": ")
            if len(parts) == 2 and parts[0] == "Relative weight":
                sensor_value = float(parts[1])

            timestamp = time.time()
            print("Sensor Value:", sensor_value)

            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

            cur.execute("INSERT INTO measurement_values (measurement_id, measurement_time, sensor_value, exp_id, machine_phase) VALUES (?, ?, ?, ?, ?)",
                        (measurement_id, timestamp_str, sensor_value, exp_id, machine_phase))
            conn.commit()

            measurement_id += 1

            # Check if the 'x' key is pressed to end the experiment
            if keyboard.is_pressed('x'):
                print("Experiment ended by user.")
                break

            # Check if the 'p' key is pressed to change the machine phase
            if keyboard.is_pressed('p'):

                # Ask the user to input the machine phase using numeric values
                new_machine_phase= input("Enter machine phase (1: not_running, 2: without_fabric, 3: with_fabric): ")
                if new_machine_phase == '1':
                    machine_phase = 'not_running'
                elif new_machine_phase == '2':
                    machine_phase = 'without_fabric'
                elif new_machine_phase == '3':
                    machine_phase = 'with_fabric'
                else:
                    print("Invalid input. Setting machine phase to not_running.")
                    machine_phase = 'not_running'
                
                # Flush out existing values in the serial buffer when 'p' key is pressed
                flush_serial_buffer(ser)
                
                new_machine_phase = machine_phase

    ser.close()

    new_experiment = input("Do you want to start a new experiment? (yes/no): ").lower()
    if new_experiment != 'yes':
        break

conn.close()

