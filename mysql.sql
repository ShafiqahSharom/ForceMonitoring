-- Create the material_properties table
CREATE TABLE IF NOT EXISTS material_properties (
    material_id INTEGER PRIMARY KEY,
    material_type TEXT,
    layer INTEGER,
    QMP_id INTEGER
);

-- Create the machine_parameter table
CREATE TABLE IF NOT EXISTS machine_parameter(
    machineparameter_id INTEGER PRIMARY KEY, 
    RPM INTEGER,
    needle_depth INTEGER,
    needle_density TEXT
);

-- Create the experiment table
CREATE TABLE IF NOT EXISTS experiment(
    exp_id INTEGER PRIMARY KEY, 
    exp_name TEXT,
    material_id INTEGER,
    machineparameter_id INTEGER,
    FOREIGN KEY (material_id) REFERENCES material(material_id) ON DELETE SET NULL,
    FOREIGN KEY (machineparameter_id) REFERENCES machine_parameter(machineparameter_id) ON DELETE SET NULL
);

-- Create the measurement_values table
CREATE TABLE IF NOT EXISTS measurement_values(
    measurement_id INTEGER, 
    measurement_time TIMESTAMP,
    sensor_value REAL, 
    machine_phase TEXT,
    exp_id INTEGER,
    PRIMARY KEY (measurement_id, exp_id),
    FOREIGN KEY (exp_id) REFERENCES experiment(exp_id) ON DELETE SET NULL
);

--Create sensor value to force value table
CREATE TABLE IF NOT EXISTS sensor_calibration(
    exp_id TEXT, 
    machine_phase TEXT, 
    mean_value REAL, 
    std_dev REAL, 
    percentage_error REAL, 
    weight_gram REAL
);

--Create sensor calibration table
CREATE TABLE IF NOT EXISTS sensor_cal(
    exp_id TEXT, 
    weight_gram REAL, 
    mean_ sensor_value REAL, 
    std_dev REAL, 
    CV REAL, 
    force_newton REAL   
);

-- Create the tensile_strength table
CREATE TABLE IF NOT EXISTS tensile_strength (
    exp_id INTEGER,
    Fmax1 REAL,
    Fmax2 REAL,
    Fmax3 REAL,
    Fmax4 REAL,
    Fmax5 REAL,
    PRIMARY KEY (exp_id),
    FOREIGN KEY (exp_id) REFERENCES experiment(exp_id) ON DELETE SET NULL
);












