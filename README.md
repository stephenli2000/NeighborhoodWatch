- [1. Block Party Registration Tools](#1-block-party-registration-tools)
  - [1.1. Setup \& Environment Installation](#11-setup--environment-installation)
    - [1.1.1. (Optional) Install a Python Version with `pyenv`](#111-optional-install-a-python-version-with-pyenv)
    - [1.1.2. Create and Activate a Virtual Environment (`venv`)](#112-create-and-activate-a-virtual-environment-venv)
    - [1.1.3. Install Dependencies](#113-install-dependencies)
  - [1.2. Generate Check-In Form](#12-generate-check-in-form)
    - [1.2.1. Extract RSVP Data (`extract_rsvp.py`)](#121-extract-rsvp-data-extract_rsvppy)
    - [1.2.2. Generate Check-In Form (`generate_check_in_form.py`)](#122-generate-check-in-form-generate_check_in_formpy)
    - [1.2.3. Output](#123-output)
  - [1.3. Generate Block Map](#13-generate-block-map)
    - [1.3.1. Download the Parcels Manually](#131-download-the-parcels-manually)
    - [1.3.2. Map Blocks to Addresses Manually](#132-map-blocks-to-addresses-manually)
    - [1.3.3. Generate the Block Map Automatically](#133-generate-the-block-map-automatically)

# 1. Block Party Registration Tools

This repository provides Python scripts to extract RSVP data from Excel workbooks, generate formatted check-in/registration tables, and create neighborhood block maps.

## 1.1. Setup & Environment Installation

### 1.1.1. (Optional) Install a Python Version with `pyenv`
If you use `pyenv` to manage Python versions, install and select your desired version (for example, Python 3.11):
```bash
pyenv install 3.11.0
pyenv local 3.11.0
```

### 1.1.2. Create and Activate a Virtual Environment (`venv`)

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

### 1.1.3. Install Dependencies
With your virtual environment active, install the required packages:
```bash
pip install -r requirements.txt
```

---

## 1.2. Generate Check-In Form

### 1.2.1. Extract RSVP Data (`extract_rsvp.py`)
Extracts worksheets from an Excel file, displays their contents in the console, and saves each worksheet as an individual `.csv` file.

```bash
# Run with default file ("Block Party Master Final.xlsx")
python extract_rsvp.py

# Run with a custom Excel file path
python extract_rsvp.py path/to/your_file.xlsx
```

### 1.2.2. Generate Check-In Form (`generate_check_in_form.py`)
Generates a formatted registration table Excel file (`.xlsx`) from a CSV file.

```bash
# Generate registration table (default output: Registration_Table_List.xlsx)
python generate_check_in_form.py input.csv

# Specify custom output filename
python generate_check_in_form.py input.csv -o Custom_Registration_List.xlsx
```

---

### 1.2.3. Output

- **`extract_rsvp.py`**: Exported `.csv` file for each worksheet in the workbook.
- **`generate_check_in_form.py`**: Formatted `.xlsx` file configured with headers, column widths, and custom row styling for event check-in.

## 1.3. Generate Block Map
### 1.3.1. Download the Parcels Manually

Download the parcel shapefile from https://data.sccgov.org/dataset/Parcels/h53q-4i8r. Click **Export**, select **Shapefile**, and unzip the downloaded archive. The extracted files will look similar to the following:

```text
-rw-rw-r-- 1          5  8月 20 21:14 geo_export_b641fa01-0ffc-4399-b325-bd4ca4f3f9db.cpg
-rw-rw-r-- 1 1750252070  8月 20 21:12 geo_export_b641fa01-0ffc-4399-b325-bd4ca4f3f9db.dbf
-rw-rw-r-- 1        235  8月 20 21:15 geo_export_b641fa01-0ffc-4399-b325-bd4ca4f3f9db.prj
-rw-rw-r-- 1  160967144  8月 20 21:14 geo_export_b641fa01-0ffc-4399-b325-bd4ca4f3f9db.shp
-rw-rw-r-- 1    3999532  8月 20 21:15 geo_export_b641fa01-0ffc-4399-b325-bd4ca4f3f9db.shx
```

### 1.3.2. Map Blocks to Addresses Manually
Calibrate the map:
```bash
python3 make_calibration.py data/saratoga.png
```
This generates `calibration.json` and `calibration_debug.png`.

Generate block addresses:
```bash
python3 saratoga_block_addresses.py data/saratoga.png --calibration calibration.json --margin-m 75
```
This generates `addresses_by_block.csv`, `blocks.json`, and `block_debug.png`.

Convert `blocks.json` to `blocks.xlsx`:
```bash
python3 blocks_to_excel.py blocks.json
```

### 1.3.3. Generate the Block Map Automatically
Process the blocks once (slow step):
```bash
python3 map_process_blocks.py -b blocks.json -s data/geo_export_b641fa01-0ffc-4399-b325-bd4ca4f3f9db.shp -o map_cache.pkl
```

This generates `map_cache.pkl`.

Render the map from the cache (fast step):
```bash
python3 map_render.py map_cache.pkl --building-fill "#A9C7F5" --building-edge "#A9C7F5" --street-font-size 12 --block-number-font-size 15 --outer-boundary-linewidth 6 --gray-outside-boundary --house-number-font-size 7
python3 map_render.py map_cache.pkl --building-fill "#5F8FD9" --building-edge "#5F8FD9" --street-font-size 12 --block-number-font-size 15 --outer-boundary-linewidth 6 --gray-outside-boundary --no-house-numbers --output map_no_house_number.png
```

Optionally, retry downloading OSM road centerlines:
```bash
python3 map_render.py map_cache.pkl --refresh-roads --street-font-size 12 --block-number-font-size 15 --house-number-font-size 7
```
