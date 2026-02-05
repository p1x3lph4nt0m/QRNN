import math as m
import pandas as pd
import numpy as np

def Accuracy(zhibiao, raw_data, raw_predictions, n):

    Data = raw_data
    test_iterations = len(Data) - n - 1
    Ei_2_sum = 0

    for j in range(test_iterations):

        if j % 20 == 0:
            print(f"[{zhibiao}] Testing step {j}/{test_iterations}")
        X_t = np.array(Data[j:j+n+2]) 
        prev_value = X_t[n]            
        Y_prediction = raw_predictions[j] 

        Ei = 0 if prev_value == 0 else m.fabs(prev_value - Y_prediction) / prev_value
        Ei_2_sum += Ei * Ei

    accuracy = 1 - m.sqrt(Ei_2_sum / test_iterations)
    print(f"Accuracy for {zhibiao}: {accuracy:.4f}")
    return accuracy

def load_predictions_from_txt(filepath):
    return np.loadtxt(filepath)

if __name__ == '__main__':

    n = 7  # must match model

    test_df = pd.read_csv("datasets/test_weather.csv")

    prediction_files = [
        "./predictions1/Atmospheric_Pressure_predictions.txt",
        "./predictions1/Minimum_Temperature_predictions.txt",
        "./predictions1/Maximum_Temperature_predictions.txt",
        "./predictions1/Relative_Humidity_predictions.txt",
        "./predictions1/Wind_Speed_predictions.txt"
    ]

    test_columns = test_df.columns.tolist()[1:]
    column_to_file = dict(zip(test_columns, prediction_files))

    for col in test_columns:
        pred_file = column_to_file[col]

        try:
            predictions = load_predictions_from_txt(pred_file)
        except Exception as e:
            print(f"Skipping {col}: cannot read {pred_file} ({e})")
            continue

        raw_values = test_df[col].values

        acc = Accuracy(
            zhibiao=col,
            raw_data=raw_values,
            raw_predictions=predictions,
            n=n
        )

        print(f"{col} accuracy: {acc:.4f}\n")
