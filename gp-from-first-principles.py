from scipy.io import loadmat
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

developer = False


def load_data(start = 0, length = 65536):

    assert length <= 65536, "Length must be less than or equal to 65536"

    #data collected during MEC326
    df1 = pd.read_csv('datasets/input.csv', header=None)
    df2 = pd.read_csv('datasets/output.csv', header=None)
    df3 = pd.read_csv('datasets/time.csv', header=None)

    df1 = df1[start:start+length]
    df2 = df2[start:start+length]
    df3 = df3[start:start+length]

    final_df = pd.concat([df1, df2, df3], axis=1)

    # Assign column names
    final_df.columns = ['input', 'output', 'time']

    # Display the final DataFrame
    print(final_df.head())
    return final_df

def plot_data(data):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.scatter(data['time'], data['input'], label='Input', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Input')
    plt.title('Input over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
    plt.scatter(data['time'], data['output'], label='Output', color='green')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.title('Output over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjusts subplot params for better layout
    plt.show()


def periodic_kernel(X1, X2, sigma, l, p):

    X1 = np.array(X1)
    X2 = np.array(X2)

    # Compute the pairwise differences between input vectors
    delta_X = X1[:, None, :] - X2[None, :, :]

    # Compute the periodic kernel matrix
    K = sigma ** 2 * np.exp(
        -2 * np.sin(np.pi * np.abs(delta_X) / p) ** 2 / l ** 2)

    return K

def main():
    data = load_data(1000, 1000)
    if developer == True: print(data)

    plot_data(data)

    input_data = np.array(data['input']).reshape(-1, 1)
    periodic_kernel_test = periodic_kernel(input_data,input_data, sigma = 10, l = 8E-4, p = 8E-4 )
    print(periodic_kernel_test)


if __name__ == "__main__":
    main()