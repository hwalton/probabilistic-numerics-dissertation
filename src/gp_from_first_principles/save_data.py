import numpy as np
import os

def save_data(start=5000, length=100):
    try:
        assert length <= 65536, "Length must be less than or equal to 65536"

        # Check current working directory
        print("Current Working Directory:", os.getcwd())

        # Data collected during MEC326
        force_response = np.loadtxt('/home/harvey/Git/probabilistic-numerics-dissertation/datasets/output.csv', delimiter=',')
        time = np.loadtxt('/home/harvey/Git/probabilistic-numerics-dissertation/datasets/time.csv', delimiter=',')

        force_response = force_response[start:start+length]
        time_truncated = time[start:start+length]

        # Add Gaussian noise to output
        # Uncomment the next line if you want to add Gaussian noise to the output
        # output = output + np.random.normal(0, 5, output.shape)

        # Save to CSV files
        np.savetxt('../../datasets/force_response.csv', force_response, delimiter=',')
        np.savetxt('../../datasets/time_truncated.csv', time_truncated, delimiter=',')

        print("Files saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    save_data()