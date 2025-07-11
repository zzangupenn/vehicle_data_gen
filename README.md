# vehicle_data_gen

## Modifications
- Compatible to current f110-gym dynamic dev branch
- Use the F1tenth Planning - MPPI package for the controller.

## `data_process.py`

This script processes and prepares vehicle dynamics data for training machine learning models. Here’s a summary of its main steps:

1. **Imports & Setup**:  
   - Imports libraries (`matplotlib`, `numpy`) and custom utilities (`DataProcessor`, `ConfigYAML`, `Logger`).
   - Sets up data directories, parameters (like `TRAIN_SEGMENT`, `TIME_INTERVAL`), and initializes a logger.

2. **Data Loading**:  
   - Defines a list of velocities (`vlist`) and friction coefficients (`flist`).
   - For each friction and velocity, loads corresponding state and control `.npy` files from disk.
   - Aggregates all loaded states and controls into arrays.

3. **Feature Selection**:  
   - Selects specific state features (columns 2, 3, 5, 6) for further processing.

4. **Normalization Parameter Calculation**:  
   - Computes normalization parameters for each selected state feature and for the computed dynamics and controls, using the `DataProcessor`.

5. **Dynamics Calculation**:  
   - Calculates the time derivative (difference divided by `TIME_INTERVAL`) of the selected state features to represent system dynamics.

6. **Save Normalization Config**:  
   - Stores normalization parameters in a YAML config file for later use.

7. **Training Data Preparation**:  
   - Segments the data into sequences of length `TRAIN_SEGMENT`.
   - Computes dynamics for each segment.
   - Optionally normalizes data (commented out).
   - Collects states, controls, dynamics, and labels for each friction setting.

8. **Save Processed Data**:  
   - Saves the processed arrays (`train_states`, `train_controls`, `train_dynamics`, `train_labels`) into a `.npz` file for training.

**In summary:**  
The script loads raw vehicle state/control data, computes dynamics, calculates normalization parameters, segments the data for training, and saves everything in a structured format for machine learning model development.

The script saves the processed data in a `.npz` file (NumPy compressed archive) with the following structure:

- **File:**  
  `train_data.npz` (or with a suffix if `SAVE_NAME` is set)

- **Contents:**  
  The file contains four arrays:

  1. **train_states**  
     - Shape: `(num_friction, num_samples, TRAIN_SEGMENT, 4)`  
     - Description: Segmented state sequences for each friction value.  
     - Each segment contains `TRAIN_SEGMENT` consecutive time steps, and each state has 4 selected features (columns 2, 3, 5, 6 from the original state array).

  2. **train_controls**  
     - Shape: `(num_friction, num_samples, 1, 2)`  
     - Description: Control inputs corresponding to each state segment.  
     - Only the first control in each segment is kept, with 2 control features.

  3. **train_dynamics**  
     - Shape: `(num_friction, num_samples, TRAIN_SEGMENT-1, 4)`  
     - Description: Dynamics (time derivatives) of the state features for each segment.

  4. **train_labels**  
     - Shape: `(num_friction, num_samples)`  
     - Description: Label indicating the friction index for each sample.

**Example:**
```
train_states:      (1, N, 2, 4)
train_controls:    (1, N, 1, 2)
train_dynamics:    (1, N, 1, 4)
train_labels:      (1, N)
```
Where `1` is the number of friction values (if only one in `flist`), and `N` is the number of segments per friction.

**Summary:**  
The output `.npz` file contains arrays for segmented states, controls, computed dynamics, and labels, all organized by friction setting and ready for use in model training.