import math as m
import pandas as pd
import numpy as np


def Accuracy_from_file(zhibiao, true_values, predictions):
    test_iterations = min(len(true_values), len(predictions))
    Ei_2_sum = 0
    
    for j in range(test_iterations):
        if j % 20 == 0:
            print(f"[{zhibiao}] Testing step {j}/{test_iterations}")

        true_val = true_values[j]
        pred_val = predictions[j]

        Ei = 0 if true_val == 0 else m.fabs(true_val - pred_val) / true_val
        Ei_2_sum += Ei * Ei

    accuracy = 1 - m.sqrt(Ei_2_sum / test_iterations)
    print(f"Accuracy for {zhibiao}: {accuracy:.4f}")

    return accuracy


def load_predictions_from_txt(filepath):
    """Load one prediction per line."""
    return np.loadtxt(filepath)


if __name__ == '__main__':
    test_df = pd.read_csv("datasets/test_weather.csv")

    prediction_files = [
        "./predictions/Atmospheric_Pressure_predictions.txt",
        "./predictions/Minimum_Temperature_predictions.txt",
        "./predictions/Maximum_Temperature_predictions.txt",
        "./predictions/Relative_Humidity_predictions.txt",
        "./predictions/Wind_Speed_predictions.txt"
    ]

    test_columns = test_df.columns.tolist()[1:]

    if len(prediction_files) != len(test_columns):
        raise ValueError("Number of prediction files must match number of test columns")

    # Map column → file
    column_to_file = dict(zip(test_columns, prediction_files))

    for col in test_columns:
        pred_file = column_to_file[col]

        try:
            predictions = load_predictions_from_txt(pred_file)
        except Exception as e:
            print(f"Skipping {col}: cannot read {pred_file} ({e})")
            continue

        true_values = test_df[col].values

        acc = Accuracy_from_file(
            zhibiao=col,
            true_values=true_values,
            predictions=predictions,
        )

        print(f"{col} accuracy: {acc:.4f}\n")
