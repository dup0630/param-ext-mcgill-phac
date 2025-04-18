import pandas as pd
import math
import sys

def safe_divide(numerator, denominator):
    """Performs safe division. Returns NaN if denominator is 0."""
    return numerator / denominator if denominator != 0 else float('nan')


def load_confusion_data(csv_path="CFR_measles.csv"):
    """Loads the main dataframe and prompts the user to select an iteration to evaluate."""
    df = pd.read_csv(csv_path)

    iteration_input = input("Enter the iteration number to evaluate: ").strip()
    try:
        iteration = int(iteration_input)
    except ValueError:
        print("Invalid iteration number. Please enter an integer.")
        sys.exit()

    df_current = df[df["Iteration"] == iteration]
    df_prev = df[df["Iteration"] == (iteration - 1)]

    if df_current.empty:
        print(f"No rows found for iteration {iteration}.")
        sys.exit()

    compare_iterations = not df_prev.empty

    if not compare_iterations:
        print(f"No rows found for previous iteration {iteration - 1}. Comparison skipped.")

    return df_current, df_prev, iteration, compare_iterations


def compute_confusion_matrix(df_current, iteration):
    """Calculates confusion matrix metrics and prints the results."""
    confusion_counts = df_current["Confusion"].value_counts().to_dict()

    TP = confusion_counts.get("TP", 0)
    TN = confusion_counts.get("TN", 0)
    FP = confusion_counts.get("FP", 0)
    FN = confusion_counts.get("FN", 0)

    print(f"\nConfusion Matrix Counts for Iteration {iteration}:")
    print(f"TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}\n")

    sensitivity = safe_divide(TP, TP + FN)
    specificity = safe_divide(TN, TN + FP)
    precision = safe_divide(TP, TP + FP)
    accuracy = safe_divide(TP + TN, TP + TN + FP + FN)
    f1_score = safe_divide(2 * TP, 2 * TP + FP + FN)
    denominator_mcc = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc = safe_divide((TP * TN - FP * FN), denominator_mcc)

    print("Evaluation Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity:          {specificity:.3f}")
    print(f"Precision (PPV):      {precision:.3f}")
    print(f"Accuracy:             {accuracy:.3f}")
    print(f"F1-score:             {f1_score:.3f}")
    print(f"Matthews CC (MCC):    {mcc:.3f}")


def analyze_iteration_changes(df_current, df_prev):
    """Compares current iteration to previous and lists changes in success/failure status."""
    prev_status = df_prev[["Paper Number", "Success/Fail"]].dropna()
    curr_status = df_current[["Paper Number", "Success/Fail"]].dropna()

    merged = pd.merge(prev_status, curr_status, on="Paper Number", suffixes=("_prev", "_curr"))

    fail_to_success = merged[(merged["Success/Fail_prev"] == "Fail") & (merged["Success/Fail_curr"] == "Success")]
    success_to_fail = merged[(merged["Success/Fail_prev"] == "Success") & (merged["Success/Fail_curr"] == "Fail")]

    print("\nChanges between previous and current iteration:")

    if not fail_to_success.empty:
        print("\nPapers that changed from Fail → Success:")
        print(fail_to_success["Paper Number"].to_list())
    else:
        print("\nNo papers changed from Fail to Success.")

    if not success_to_fail.empty:
        print("\nPapers that changed from Success → Fail:")
        print(success_to_fail["Paper Number"].to_list())
    else:
        print("\nNo papers changed from Success to Fail.")


if __name__ == "__main__":
    df_current, df_prev, iteration, compare_iterations = load_confusion_data()
    compute_confusion_matrix(df_current, iteration)

    if compare_iterations:
        analyze_iteration_changes(df_current, df_prev)
