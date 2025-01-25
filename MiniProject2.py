import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_mean_erp(trial_points, ecog_data):
    # Load trial points with appropriate column names
    data = pd.read_csv(trial_points, header=None, names=['starting_point', 'peak_point', 'finger'])
    
    # Ensure data types are int for trial_points
    trial_points_int = data.apply(pd.to_numeric, errors='coerce').dropna().astype(int)

    # Load the ECOG data
    ecog_data = pd.read_csv(ecog_data, header=None).to_numpy(dtype=np.float64).flatten()

    # Time window constants
    pre_start, post_start = 200, 1000
    window_length = pre_start + post_start + 1

    # Store ERPs by finger
    fingers_erp = {finger: [] for finger in range(1, 6)}

    # Extract segments for each event
    for _, row in trial_points_int.iterrows():
        start_idx, finger = row['starting_point'], row['finger']
        if 0 <= start_idx - pre_start < len(ecog_data) - post_start:
            segment = ecog_data[start_idx - pre_start:start_idx + post_start + 1]
            fingers_erp[finger].append(segment)

    # Calculate mean ERPs
    fingers_erp_mean_matrix = np.array([
        np.mean(fingers_erp[finger], axis=0) if fingers_erp[finger] else np.zeros(window_length)
        for finger in range(1, 6)
    ], dtype=np.float64)

    # Create DataFrame for mean ERPs (optional, can be removed)
    fingers_erp_mean = pd.DataFrame(fingers_erp_mean_matrix, index=[str(finger) for finger in range(1, 6)], columns=np.arange(window_length))

    # Print the mean ERPs DataFrame
    print(fingers_erp_mean)

    # Plot the mean ERPs
    plt.figure(figsize=(12, 6))
    for finger in range(1, 6):
        plt.plot(fingers_erp_mean.columns, fingers_erp_mean.loc[str(finger)], label=f'Finger {finger}')
    
    plt.title('Mean ERPs for Each Finger')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (μV)')
    plt.axvline(x=pre_start, color='gray', linestyle='--', label='Stimulus Onset')
    plt.legend()
    plt.grid()
    plt.show()

    return fingers_erp_mean_matrix  # return the matrix directly

# Usage
trial_points = r"C:\\Users\\linoy\\OneDrive\\שולחן העבודה\\MiniProject Python\\mini_project_2_data\\events_file_ordered.csv"
ecog_data_file = r"C:\\Users\\linoy\\OneDrive\\שולחן העבודה\\MiniProject Python\\mini_project_2_data\\brain_data_channel_one.csv"
fingers_erp_mean = calc_mean_erp(trial_points, ecog_data_file)

