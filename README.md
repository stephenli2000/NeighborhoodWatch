# Block Party Registration Tools

This repository provides Python scripts to extract RSVP data from Excel workbooks and generate formatted check-in/registration tables for block parties.

## Setup & Environment Installation

### 1. (Optional) Install Python Version with `pyenv`
If you use `pyenv` to manage Python versions, set up your desired version (e.g., Python 3.11):
```bash
pyenv install 3.11.0
pyenv local 3.11.0
```

### 2. Create and Activate a Virtual Environment (`venv`)

**On macOS / Linux:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows (Command Prompt / PowerShell):**
```cmd
:: Command Prompt
python -m venv venv
venv\Scripts\activate.bat

# PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
With your virtual environment active, install the required packages:
```bash
pip install -r requirements.txt
```

---

## Usage

### Extract RSVP Data (`extract_rsvp.py`)
Extracts sheets from an Excel file, displays contents in the console, and saves each sheet as an individual `.csv` file.

```bash
# Run with default file ("Block Party Master Final.xlsx")
python extract_rsvp.py

# Run with a custom Excel file path
python extract_rsvp.py path/to/your_file.xlsx
```

### Generate Check-In Form (`generate_check_in_form.py`)
Generates a formatted registration table Excel file (`.xlsx`) from a CSV file.

```bash
# Generate registration table (default output: Registration_Table_List.xlsx)
python generate_check_in_form.py input.csv

# Specify custom output filename
python generate_check_in_form.py input.csv -o Custom_Registration_List.xlsx
```

---

## Output

- **`extract_rsvp.py`**: Exported `.csv` file for each worksheet in the workbook.
- **`generate_check_in_form.py`**: Formatted `.xlsx` file configured with headers, column widths, and custom row styling for event check-in.
