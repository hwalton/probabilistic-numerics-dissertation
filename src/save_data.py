import numpy as np
import os
import dotenv
dotenv.load_dotenv()

from jax.random import PRNGKey, split, normal


def save_data(length = 512, dataset = 4):
    assert length % 2 == 0, "Length must be even"
    try:
        if dataset == 0:
            start = 5000
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
        elif dataset == 1:
            key = PRNGKey(0)
            sn2 = 1

            time_truncated = np.linspace(0, 6 * np.pi, length)[:, None]
            force_response = 2 * np.sin(10 * time_truncated) + sn2 * normal(key, shape=time_truncated.shape)
        elif dataset == 2:
            key = PRNGKey(0)
            sn2 = 0

            time_truncated = np.linspace(0, 10, length)[:, None]
            force_response = 3 * np.sin(5 * time_truncated + 0.2) + sn2 * normal(key, shape=time_truncated.shape)
        elif dataset == 3:
            key = PRNGKey(0)
            sn2 = 0

            time_truncated = np.linspace(0, 25, length)[:, None]
            force_response = 1 * np.sin(3 * time_truncated + 0.2) + sn2 * normal(key, shape=time_truncated.shape) + \
                             2 * np.sin(10 * time_truncated + 2) + sn2 * normal(key, shape=time_truncated.shape) +  \
                             3 * np.sin(5 * time_truncated + 3) + sn2 * normal(key, shape=time_truncated.shape)

        elif dataset == 4:
            key = PRNGKey(0)
            sn2 = 0.25
            m = float(os.getenv('M'))  # Mass
            c = float(os.getenv('C'))  # Damping coefficient
            k = float(os.getenv('K'))  # Stiffness
            sample_rate = 32

            # Time array
            time_truncated = np.linspace(0, (length-1)/sample_rate, length)[:, None]

            # Calculate natural frequency and damping ratio
            omega_n = np.sqrt(k / m)
            zeta = c / (2 * np.sqrt(m * k))

            # Calculate damped natural frequency
            omega_d = omega_n * np.sqrt(1 - zeta ** 2)

            # Assume A=1 and phi=0 for simplicity, these should be determined based on initial conditions
            A = 1
            phi = 0

            # Calculate the force response (displacement response) of the system
            force_response = A * np.exp(-zeta * omega_n * time_truncated) * np.cos(omega_d * time_truncated + phi) + sn2 * normal(key, shape=time_truncated.shape)

        # Save to CSV files
        np.savetxt('../datasets/force_response.csv', force_response, delimiter=',')
        np.savetxt('../datasets/time_truncated.csv', time_truncated, delimiter=',')

        print("Files saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    save_data()