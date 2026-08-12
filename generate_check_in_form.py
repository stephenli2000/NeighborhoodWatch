import argparse
import os
import sys
import openpyxl
import pandas as pd
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

def generate_registration_table(input_csv, output_xls):
    # Read input CSV
    df = pd.read_csv(input_csv)

    # Detect relevant columns by keywords or fallback to default positions
    street_col = [c for c in df.columns if "STREET" in c.upper()]
    names_col = [
        c for c in df.columns if "NAMES" in c.upper() or "ATTENDING" in c.upper()
    ]
    headcount_col = [
        c
        for c in df.columns
        if "HEADCOUNT" in c.upper() or "NUMBER" in c.upper()
    ]
    zone_col = [c for c in df.columns if "ZONE" in c.upper()]

    col_street = street_col[0] if street_col else df.columns[2]
    col_names = names_col[0] if names_col else df.columns[7]
    col_headcount = headcount_col[0] if headcount_col else df.columns[8]
    col_zone = zone_col[0] if zone_col else df.columns[12]

    # Create Workbook and worksheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Regist Table List"
    
    # Define Styles based on screenshot
    font_title = Font(name="Calibri", size=14, bold=True)
    font_header = Font(name="Calibri", size=9, bold=True, color="FFFFFF") # White text for headers
    font_regular = Font(name="Calibri", size=10)
    
    # Header row fill is black
    header_fill = PatternFill(start_color="000000", end_color="000000", fill_type="solid")
    
    # Alternating row fills
    white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    gray_fill = PatternFill(start_color="EAEAEA", end_color="EAEAEA", fill_type="solid")
    
    # Green highlight for Block Captains
    green_fill = PatternFill(start_color="B4E0A2", end_color="B4E0A2", fill_type="solid")
    
    # Thin borders for all cells in the table
    thin_border = Border(left=Side(style='thin'), 
                         right=Side(style='thin'), 
                         top=Side(style='thin'), 
                         bottom=Side(style='thin'))

    # Set Column Widths 
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 8
    ws.column_dimensions['C'].width = 10
    ws.column_dimensions['D'].width = 10
    ws.column_dimensions['E'].width = 35
    ws.column_dimensions['F'].width = 65
    ws.column_dimensions['G'].width = 8
    ws.column_dimensions['H'].width = 45
    ws.column_dimensions['I'].width = 8

    # Header / Metadata Layout
    ws.cell(row=1, column=8, value="=TODAY()").font = Font(name="Calibri", size=10)
    ws.cell(row=1, column=8).alignment = Alignment(horizontal='right')
    
    cell_title = ws.cell(row=2, column=5, value="Registration Table List")
    cell_title.font = font_title
    
    ws.cell(row=2, column=6, value="Sorted by Street Address Number").font = Font(name="Calibri", size=10)
    ws.cell(row=2, column=6).alignment = Alignment(horizontal='right')
    ws.cell(row=2, column=11, value="Addresses with 0 are hidden").font = Font(name="Calibri", size=10)
    
    # Notice cell in green
    cell_notice = ws.cell(row=3, column=8, value="Ask if they're willing to be their Block Captain")
    cell_notice.fill = green_fill
    cell_notice.font = font_regular
    cell_notice.border = thin_border

    # Column Headers (Row 4)
    headers = [
        "email",
        "Flyer",
        "Postcard",
        "Ted's sign",
        "Street Address",
        "Names of Attendees",
        "Number",
        "Zone",
        "Row",
    ]
    for c_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=4, column=c_idx, value=h)
        cell.font = font_header
        cell.fill = header_fill
        cell.border = thin_border
        
        # Angled text for first 4 columns
        if c_idx <= 4:
            cell.alignment = Alignment(text_rotation=45, horizontal='center', vertical='bottom')
        else:
            cell.alignment = Alignment(vertical='bottom')

    # Populate table rows
    row_idx = 5
    for idx, row in df.iterrows():
        street = str(row[col_street]) if pd.notna(row[col_street]) else ""
        names = str(row[col_names]) if pd.notna(row[col_names]) else ""
        num = row[col_headcount] if pd.notna(row[col_headcount]) else ""
        zone = str(row[col_zone]) if pd.notna(row[col_zone]) else ""

        # Alternating background colors
        row_fill = gray_fill if row_idx % 2 == 0 else white_fill

        cells = [
            ws.cell(row=row_idx, column=1, value=""),
            ws.cell(row=row_idx, column=2, value=""),
            ws.cell(row=row_idx, column=3, value=""),
            ws.cell(row=row_idx, column=4, value=""),
            ws.cell(row=row_idx, column=5, value=street),
            ws.cell(row=row_idx, column=6, value=names),
            ws.cell(row=row_idx, column=7, value=num),
            ws.cell(row=row_idx, column=8, value=zone),
            ws.cell(row=row_idx, column=9, value=row_idx),
        ]
        
        for i, cell in enumerate(cells):
            cell.font = font_regular
            cell.border = thin_border
            
            # Apply standard row fill
            if i != 7: # Skip Zone column for now
                cell.fill = row_fill
                
            # Right align Numbers and Row indices
            if i == 6 or i == 8: 
                cell.alignment = Alignment(horizontal='right')

        # Formatting for the Zone column (Green if "Will you", otherwise alternating row color)
        if "Will you" in zone:
            cells[7].fill = green_fill
        else:
            cells[7].fill = row_fill

        row_idx += 1

    # Save to Excel file
    wb.save(output_xls)
    print(f"Successfully generated '{output_xls}' with matching design from screenshot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Registration Table List Excel file from CSV input."
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument(
        "-o",
        "--output",
        default="Registration_Table_List.xlsx",
        help="Output Excel filename (.xlsx)",
    )

    args = parser.parse_args()
    generate_registration_table(args.input_csv, args.output)