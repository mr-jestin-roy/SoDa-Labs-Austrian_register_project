"""
merge_pages_to_master.py
───────────────────────
Merge all page_XXXX_ocr.xlsx files into single master XLSX per directory

For each RTB directory:
  - Reads all page_XXXX_ocr.xlsx files
  - Extracts data rows (skips headers)
  - Sorts by batch, then cells (ascending)
  - Writes to master_RTB_X.xlsx

Output:
  - master_RTB_1_Aach-Berlichingen.xlsx
  - master_RTB_2_Berlin-Dotterwies.xlsx
  - master_RTB_3_Drachhausen-Kaprun.xlsx
  - master_RTB_4_Karbow-Quittainen.xlsx

Run:
    python merge_pages_to_master.py
"""

import glob
import os
from pathlib import Path
from collections import defaultdict

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    import pandas as pd
except ImportError:
    import sys
    sys.exit("Missing: openpyxl, pandas  →  pip install openpyxl pandas")

BASE = Path(__file__).parent

RTB_CONFIGS = [
    {
        'dir': BASE / "results_RTB_1_Aach-Berlichingen",
        'output': BASE / "master_RTB_1_Aach-Berlichingen.xlsx",
        'name': "RTB_1_Aach-Berlichingen"
    },
    {
        'dir': BASE / "results_RTB_2_Berlin-Dotterwies",
        'output': BASE / "master_RTB_2_Berlin-Dotterwies.xlsx",
        'name': "RTB_2_Berlin-Dotterwies"
    },
    {
        'dir': BASE / "results_RTB_3_Drachhausen-Kaprun",
        'output': BASE / "master_RTB_3_Drachhausen-Kaprun.xlsx",
        'name': "RTB_3_Drachhausen-Kaprun"
    },
    {
        'dir': BASE / "results_RTB_4_Karbow-Quittainen",
        'output': BASE / "master_RTB_4_Karbow-Quittainen.xlsx",
        'name': "RTB_4_Karbow-Quittainen"
    },
    {
        'dir': BASE / "results_RTB_5_Raab-Zygry",
        'output': BASE / "master_RTB_5_Raab-Zygry.xlsx",
        'name': "RTB_5_Raab-Zygry"
    },
]

def extract_batch_cell_sort_key(batch: str, cells: str):
    """Extract sort keys from batch and cells for proper ordering."""
    # batch format: "page_0209_b02"
    # cells format: "col1_sec02.png"

    import re

    # Extract page number
    page_match = re.search(r'page_(\d+)', batch)
    page_num = int(page_match.group(1)) if page_match else 99999

    # Extract batch number
    batch_match = re.search(r'b(\d+)', batch)
    batch_num = int(batch_match.group(1)) if batch_match else 999

    # Extract col and sec from cells
    cell_match = re.match(r'col(\d+)_sec(\d+)', cells)
    if cell_match:
        col_num = int(cell_match.group(1))
        sec_num = int(cell_match.group(2))
    else:
        col_num = 999
        sec_num = 999

    return (page_num, batch_num, col_num, sec_num)

def merge_directory(config: dict) -> dict:
    """Merge all page XLSX files in a directory."""

    rtb_dir = config['dir']
    output_path = config['output']
    rtb_name = config['name']

    print(f"\n▶ Processing: {rtb_name}")
    print("-" * 80)

    if not rtb_dir.exists():
        print(f"✗ Directory not found: {rtb_dir}")
        return {'error': 'Directory not found', 'rows': 0}

    # Find all page XLSX files
    xlsx_files = sorted(glob.glob(str(rtb_dir / "page_*_ocr.xlsx")))
    print(f"  Found {len(xlsx_files)} XLSX files")

    # Read all files into dataframes
    all_rows = []
    file_count = 0
    error_count = 0

    for xlsx_file in xlsx_files:
        try:
            df = pd.read_excel(xlsx_file, sheet_name=0)
            # Skip if empty or only has header
            if len(df) == 0:
                continue
            all_rows.append(df)
            file_count += 1
        except Exception as e:
            error_count += 1
            print(f"  ✗ Error reading {Path(xlsx_file).name}: {str(e)[:50]}")

    print(f"  Successfully read: {file_count} files, {error_count} errors")

    if not all_rows:
        print(f"✗ No data found in any files")
        return {'error': 'No data found', 'rows': 0}

    # Concatenate all dataframes
    master_df = pd.concat(all_rows, ignore_index=True)
    print(f"  Total rows before sort: {len(master_df)}")

    # Sort by batch and cells (ascending)
    def sort_key(row):
        return extract_batch_cell_sort_key(
            str(row['batch']),
            str(row['cells'])
        )

    master_df['_sort_key'] = master_df.apply(
        lambda row: sort_key(row),
        axis=1
    )
    master_df = master_df.sort_values('_sort_key').drop('_sort_key', axis=1)
    master_df = master_df.reset_index(drop=True)

    print(f"  Total rows after sort: {len(master_df)}")

    # Create XLSX with formatting
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Reichstelefonbuch"

    # Header style
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Write headers
    headers = ['batch', 'cells', 'region', 'phone_extension', 'transcribed_text']
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align

    # Write data rows
    for row_idx, (_, row) in enumerate(master_df.iterrows(), start=2):
        ws.cell(row=row_idx, column=1, value=row['batch'])
        ws.cell(row=row_idx, column=2, value=row['cells'])
        ws.cell(row=row_idx, column=3, value=row['region'])
        ws.cell(row=row_idx, column=4, value=row['phone_extension'])
        ws.cell(row=row_idx, column=5, value=row['transcribed_text'])

    # Column widths
    ws.column_dimensions["A"].width = 22   # batch
    ws.column_dimensions["B"].width = 18   # cells
    ws.column_dimensions["C"].width = 30   # region
    ws.column_dimensions["D"].width = 16   # phone_extension
    ws.column_dimensions["E"].width = 80   # transcribed_text

    # Freeze header row
    ws.freeze_panes = "A2"

    # Save
    wb.save(output_path)
    print(f"  ✔ Master XLSX saved → {output_path.name}")
    print(f"     Rows: {len(master_df)}, Columns: {len(master_df.columns)}")

    return {
        'dir': rtb_name,
        'output': str(output_path),
        'rows': len(master_df),
        'files_merged': file_count,
        'errors': error_count
    }

# ── Main execution ────────────────────────────────────────────────────────────
print("=" * 80)
print("▶ MERGE: All page_XXXX_ocr.xlsx → Master XLSX per RTB directory")
print("=" * 80)

results = []
total_rows = 0

for config in RTB_CONFIGS:
    result = merge_directory(config)
    if 'error' not in result:
        results.append(result)
        total_rows += result['rows']

print("\n" + "=" * 80)
print("▶ SUMMARY")
print("=" * 80)
for result in results:
    print(f"  {result['dir']}")
    print(f"    Files merged: {result['files_merged']}")
    print(f"    Total rows:   {result['rows']}")
    print(f"    Output:       {Path(result['output']).name}")
    print()

print(f"  Grand total rows: {total_rows}")
print("=" * 80)
print("✔ COMPLETE")
print("=" * 80)
