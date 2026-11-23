#!/usr/bin/env python3
"""
Convert Chandra HTML output to XLSX files with preserved table structure.
- Retains multi-level headers with merged cells (rowspan/colspan)
- Files without _L/_R: 2 tables → Left + Right sheets
- Files with _L/_R: 1 table → single sheet
"""
import os
from pathlib import Path
from bs4 import BeautifulSoup
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("xlsx_output")

# Styles
HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
HEADER_FONT = Font(bold=True, color="FFFFFF")
THIN_BORDER = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)


def parse_html_table(table_element):
    """
    Parse an HTML table and return structured data with span information.
    Returns: (header_rows, data_rows, merges, num_cols)
    """
    # Build a grid to track cell occupancy
    grid = {}  # (row, col) -> cell_value
    merges = []  # list of (start_row, start_col, end_row, end_col)

    all_rows = table_element.find_all('tr')

    # Determine max columns by scanning all rows
    max_cols = 0
    for tr in all_rows:
        cells = tr.find_all(['th', 'td'])
        col_count = sum(int(cell.get('colspan', 1)) for cell in cells)
        max_cols = max(max_cols, col_count)

    # Parse each row
    current_row = 0
    header_row_count = 0
    header_rows = []
    data_rows = []

    for tr in all_rows:
        cells = tr.find_all(['th', 'td'])
        is_header_row = any(cell.name == 'th' for cell in cells)

        if is_header_row:
            header_row_count += 1

        current_col = 0
        row_data = []

        for cell in cells:
            # Find next available column (skip occupied cells from rowspan)
            while (current_row, current_col) in grid:
                current_col += 1

            # Get cell text (replace <br> with newlines)
            for br in cell.find_all('br'):
                br.replace_with('\n')
            cell_text = cell.get_text(strip=False).strip()

            # Get span values
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))

            # Mark all cells in the span as occupied
            for r in range(rowspan):
                for c in range(colspan):
                    grid[(current_row + r, current_col + c)] = cell_text if (r == 0 and c == 0) else ""

            # Record merge if needed
            if rowspan > 1 or colspan > 1:
                merges.append((
                    current_row + 1,  # Excel is 1-indexed
                    current_col + 1,
                    current_row + rowspan,
                    current_col + colspan
                ))

            row_data.append({
                'value': cell_text,
                'colspan': colspan,
                'rowspan': rowspan,
                'is_header': cell.name == 'th'
            })

            current_col += colspan

        if is_header_row:
            header_rows.append(row_data)
        else:
            data_rows.append(row_data)

        current_row += 1

    return header_rows, data_rows, merges, max_cols, header_row_count


def write_table_to_sheet(ws, table_element):
    """Write an HTML table to an Excel worksheet with proper structure."""
    header_rows, data_rows, merges, max_cols, header_row_count = parse_html_table(table_element)

    # Build grid from parsed data
    grid = {}
    all_rows = table_element.find_all('tr')

    current_row = 0
    for tr in all_rows:
        cells = tr.find_all(['th', 'td'])
        current_col = 0

        for cell in cells:
            # Find next available column
            while (current_row, current_col) in grid:
                current_col += 1

            # Get cell text
            for br in cell.find_all('br'):
                br.replace_with('\n')
            cell_text = cell.get_text(strip=False).strip()

            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            is_header = cell.name == 'th'

            # Mark occupied cells
            for r in range(rowspan):
                for c in range(colspan):
                    grid[(current_row + r, current_col + c)] = {
                        'value': cell_text if (r == 0 and c == 0) else "",
                        'is_header': is_header,
                        'is_merge_origin': (r == 0 and c == 0) and (rowspan > 1 or colspan > 1)
                    }

            current_col += colspan
        current_row += 1

    # Write to Excel
    for (row, col), cell_data in grid.items():
        excel_row = row + 1  # Excel is 1-indexed
        excel_col = col + 1

        cell = ws.cell(row=excel_row, column=excel_col, value=cell_data['value'])

        # Apply styling
        cell.alignment = Alignment(
            horizontal="center" if cell_data['is_header'] else "left",
            vertical="center" if cell_data['is_header'] else "top",
            wrap_text=True
        )
        cell.border = THIN_BORDER

        if cell_data['is_header']:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT

    # Apply merges
    for start_row, start_col, end_row, end_col in merges:
        ws.merge_cells(
            start_row=start_row,
            start_column=start_col,
            end_row=end_row,
            end_column=end_col
        )

    # Set column widths
    for col in range(1, max_cols + 1):
        ws.column_dimensions[get_column_letter(col)].width = 15


def convert_html_to_xlsx(html_path: Path, output_dir: Path):
    """Convert a single HTML file to XLSX with preserved structure."""
    filename = html_path.stem

    try:
        # Read HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')

        if not tables:
            print(f"⚠️  No tables found in {html_path.name}")
            return False

        xlsx_path = output_dir / f"{filename}.xlsx"
        wb = Workbook()

        # Remove default sheet
        wb.remove(wb.active)

        # Check if it's a split file (_L or _R)
        is_split = "_L" in filename or "_R" in filename

        if is_split:
            # Single table file
            ws = wb.create_sheet("Table")
            write_table_to_sheet(ws, tables[0])
            print(f"✅ {html_path.name} → 1 table")
        else:
            # Two tables (Left + Right pages)
            if len(tables) >= 2:
                ws_left = wb.create_sheet("Left_Page")
                write_table_to_sheet(ws_left, tables[0])

                ws_right = wb.create_sheet("Right_Page")
                write_table_to_sheet(ws_right, tables[1])
                print(f"✅ {html_path.name} → 2 tables (Left + Right)")
            elif len(tables) == 1:
                ws = wb.create_sheet("Table")
                write_table_to_sheet(ws, tables[0])
                print(f"⚠️  {html_path.name} → 1 table (expected 2)")

        wb.save(xlsx_path)
        return True

    except Exception as e:
        print(f"❌ Error processing {html_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all HTML files (exclude this script if somehow named .html)
    html_files = sorted([f for f in INPUT_DIR.glob("*.html") if f.is_file()])

    print(f"Found {len(html_files)} HTML files")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("-" * 60)

    success = 0
    failed = 0

    for html_file in html_files:
        if convert_html_to_xlsx(html_file, OUTPUT_DIR):
            success += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"✅ Successfully converted: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
