import numpy as np
import os
import dotenv
dotenv.load_dotenv()

from jax.random import PRNGKey, split, normal


def save_data(sample_rate=32, length=256, dataset=4, input_noise_stdv=10, response_noise_stdv=0.25):
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
            time = time[start:start+length]

            # Add Gaussian noise to output
            # Uncomment the next line if you want to add Gaussian noise to the output
            # output = output + np.random.normal(0, 5, output.shape)
        elif dataset == 1:
            key = PRNGKey(0)
            sn2 = 1

            time = np.linspace(0, 6 * np.pi, length)[:, None]
            force_response = 2 * np.sin(10 * time) + sn2 * normal(key, shape=time.shape)
        elif dataset == 2:
            key = PRNGKey(0)
            sn2 = 0

            time = np.linspace(0, 10, length)[:, None]
            force_response = 3 * np.sin(5 * time + 0.2) + sn2 * normal(key, shape=time.shape)
        elif dataset == 3:
            key = PRNGKey(0)
            sn2 = 0

            time = np.linspace(0, 25, length)[:, None]
            force_response = 1 * np.sin(3 * time + 0.2) + sn2 * normal(key, shape=time.shape) + \
                             2 * np.sin(10 * time + 2) + sn2 * normal(key, shape=time.shape) +  \
                             3 * np.sin(5 * time + 3) + sn2 * normal(key, shape=time.shape)

        elif dataset == 4:
            time = np.linspace(0, (length - 1) / sample_rate, length)

            key = PRNGKey(0)
            m = float(os.getenv('M'))  # Mass
            c = float(os.getenv('C'))  # Damping coefficient
            k = float(os.getenv('K'))  # Stiffness

            # Time array

            time += input_noise_stdv * normal(key, shape=time.shape)
            time = np.sort(time)[:, None]

            # Calculate natural frequency and damping ratio
            omega_n = np.sqrt(k / m)
            zeta = c / (2 * np.sqrt(m * k))

            # Calculate damped natural frequency
            omega_d = omega_n * np.sqrt(1 - zeta ** 2)

            # Assume A=1 and phi=0 for simplicity, these should be determined based on initial conditions
            A = 1
            phi = 0

            # Calculate the force response (displacement response) of the system
            force_response = A * omega_n ** 2 * np.exp(-zeta * omega_n * time) * np.sin(omega_d * time + phi) + response_noise_stdv * normal(key, shape=time.shape)

        elif dataset == 5:
            # time = np.random.uniform(0, (length - 1) / sample_rate, size=length)
            time = np.random.triangular(left=0, mode=0, right=(length - 1) / sample_rate, size=length)
            time = np.sort(time)[:, None]

            key = PRNGKey(0)

            phi = float(os.getenv('PHI'))

            m_2 = float(os.getenv('M_2'))
            c_2 = float(os.getenv('C_2'))
            k_2 = float(os.getenv('K_2'))

            omega_n_2 = np.sqrt(k_2 / m_2)
            zeta_2  = c_2 / (2 * np.sqrt(m_2 * k_2))

            # Calculate damped natural frequency
            omega_d_2 = omega_n_2 * np.sqrt(1 - zeta_2 ** 2)

            A_2 = float(os.getenv('A_2'))

            phi_2 = phi

            # Calculate the force response (displacement response) of the system
            force_response = A_2 * omega_n_2 ** 2 * np.exp(-zeta_2 * omega_n_2 * time) * np.sin(omega_d_2 * time + phi_2) \
                           + response_noise_stdv * normal(key, shape=time.shape)

        elif dataset == 6:
            time = np.linspace(0, (length - 1) / sample_rate, length)

            key = PRNGKey(0)
            m = float(os.getenv('M_2'))  # Mass
            c = float(os.getenv('C_2'))  # Damping coefficient
            k = float(os.getenv('K_2'))  # Stiffness

            # Time array

            time += input_noise_stdv * normal(key, shape=time.shape)
            time = np.sort(time)[:, None]

            # Calculate natural frequency and damping ratio
            omega_n = np.sqrt(k / m)
            zeta = c / (2 * np.sqrt(m * k))

            # Calculate damped natural frequency
            omega_d = omega_n * np.sqrt(1 - zeta ** 2)

            # Assume A=1 and phi=0 for simplicity, these should be determined based on initial conditions
            A = 1
            phi = 0

            # Calculate the force response (displacement response) of the system
            force_response = A * omega_n ** 2 * np.exp(-zeta * omega_n * time) * np.sin(omega_d * time + phi) + response_noise_stdv * normal(key, shape=time.shape)



        # Save to CSV files
        np.savetxt('../datasets/force_response.csv', force_response, delimiter=',')
        np.savetxt('../datasets/time_truncated.csv', time, delimiter=',')

        print("Files saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    length = 512
    sample_rate = 32

    time = np.linspace(0, (length - 1) / sample_rate, length)[:, None]
    save_data(time)