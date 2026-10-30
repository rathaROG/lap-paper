"""Update existing LaTeX numeric table cells from saved-data projections."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re


def format_numeric_cells(data: dict, row_index: int) -> list[str]:
    """Format scalar cells, ranges, and median/IQR triplets from raw numbers.

    Controlled tables use ``values`` and one scalar per cell. Application
    projections use flattened ``numeric_values`` with explicit cell widths.
    Descriptive prose cells are never used as the numerical source of truth.
    """
    values = data.get('numeric_values', data.get('values'))
    if values is None:
        raise ValueError('Projection has no numeric rows')
    row = values[row_index]
    widths = data.get('column_numeric_widths', [1] * len(row))
    places = data['places']
    digits = [places] * len(row) if isinstance(places, int) else places
    if sum(widths) != len(row) or len(digits) != len(row):
        raise ValueError('Projection cell widths/precision do not cover numeric row')
    if any(not math.isfinite(float(value)) for value in row):
        raise ValueError('Nonfinite publication table value')
    numbers = [f'{float(value):.{precision}f}' for value, precision in zip(row, digits)]
    cells, offset = [], 0
    for width in widths:
        group = numbers[offset:offset + width]
        if width == 1:
            cells.append(group[0])
        elif width == 2:
            cells.append('--'.join(group))
        elif width == 3:
            cells.append(r'\shortstack[r]{' + group[0] + r'\\{[' + group[1] + ', ' + group[2] + ']}}')
        else:
            raise ValueError(f'Unsupported numeric cell width: {width}')
        offset += width
    return cells


def apply_tables(tex: str, tables: dict) -> str:
    tables = tables.get('tables', tables)
    for label, data in tables.items():
        values = data.get('numeric_values', data.get('values'))
        if values is None:
            raise ValueError(f'Table {label}: no numeric values')
        marker = rf'\label{{tab:{label}}}'
        start = tex.index(marker)
        end = tex.index(r'\end{table}', start)
        segment = tex[start:end]
        head, body = segment.split(r'\midrule', 1)
        row_index = 0
        output = []
        for line in body.splitlines():
            if '&' in line and line.rstrip().endswith(r'\\') and not line.lstrip().startswith(r'\multicolumn'):
                cells = line.rsplit(r'\\', 1)[0].split('&')
                expected = format_numeric_cells(data, row_index)
                if len(cells) - 1 != len(expected):
                    raise ValueError(f'Table {label}, row {row_index}: column mismatch')
                line = cells[0].rstrip() + ' & ' + ' & '.join(expected) + r' \\'
                row_index += 1
            output.append(line)
        if row_index != len(values):
            raise ValueError(f'Table {label}: expected {len(values)} rows, found {row_index}')
        tex = tex[:start] + head + r'\midrule' + '\n'.join(output) + '\n' + tex[end:]
    return tex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('projections', type=Path)
    parser.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[1] / 'latex/main.tex')
    args = parser.parse_args()
    loaded = json.loads(args.projections.read_text(encoding='utf-8'))
    tables = loaded.get('tables', loaded)
    args.paper.write_text(apply_tables(args.paper.read_text(encoding='utf-8'), tables), encoding='utf-8')
    print(f'Updated {len(tables)} numeric tables from saved-data projections.')


if __name__ == '__main__':
    main()
