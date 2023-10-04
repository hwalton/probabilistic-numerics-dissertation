from scipy.io import loadmat
import os
import csv
import pandas as pd

developer = True


def load_data(start = 0, length = 65536):

    assert length <= 65536, "Length must be less than or equal to 65536"

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


def main():
    data = load_data(1000, 1000)
    if developer == True: print(data)


if __name__ == "__main__":
    main()