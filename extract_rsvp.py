import argparse
import os
import pandas as pd


def extract_rsvp(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)

    # Iterate through each sheet in the workbook
    for sheet_name in excel_file.sheet_names:
        print(f"\n==========================================")
        print(f" Reading Sheet: {sheet_name}")
        print(f"==========================================\n")

        # Read data into DataFrame
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Print data to console
        print(df)

        # Define output CSV filename
        # Sanitize sheet name for valid filename output
        clean_sheet_name = "".join(
            c if c.isalnum() or c in (" ", "_", "-") else "_" for c in sheet_name
        )
        csv_filename = f"{clean_sheet_name}.csv"

        # Save to CSV format
        df.to_csv(csv_filename, index=False)
        print(f"\n Successfully saved '{sheet_name}' to '{csv_filename}'")


def main():
    parser = argparse.ArgumentParser(
        description="Extract sheets from an Excel workbook into individual CSV files."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default="Block Party Master Final.xlsx",
        help="Path to input Excel file (default: Block Party Master Final.xlsx)",
    )

    args = parser.parse_args()
    extract_rsvp(args.file_path)


if __name__ == "__main__":
    main()