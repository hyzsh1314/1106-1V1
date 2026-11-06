"""Torch-native interpolation and JSBSim XML table loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


@dataclass(frozen=True)
class TableData:
    row: torch.Tensor
    values: torch.Tensor
    column: torch.Tensor | None = None


def _numbers(text: str | None) -> list[list[float]]:
    rows = []
    for line in (text or "").strip().splitlines():
        values = line.split()
        if values:
            rows.append([float(value) for value in values])
    return rows


def load_jsbsim_function_tables(path: str | Path) -> dict[str, TableData]:
    """Load the first lookup table under each named JSBSim function.

    XML parsing happens once during module construction.  Simulation itself
    uses only registered tensors, so the hot path remains GPU resident and is
    compatible with ``torch.compile``.
    """

    root = ET.parse(path).getroot()
    result: dict[str, TableData] = {}
    for function in root.iter("function"):
        name = function.get("name")
        table = function.find(".//table")
        if not name or table is None:
            continue
        independent = table.findall("independentVar")
        data = table.find("tableData")
        rows = _numbers(None if data is None else data.text)
        if not rows:
            continue
        if len(independent) == 1:
            tensor = torch.tensor(rows, dtype=torch.float32)
            result[name] = TableData(
                tensor[:, 0].contiguous(), tensor[:, 1].contiguous())
        elif len(independent) == 2:
            column = torch.tensor(rows[0], dtype=torch.float32)
            tensor = torch.tensor(rows[1:], dtype=torch.float32)
            result[name] = TableData(
                tensor[:, 0].contiguous(), tensor[:, 1:].contiguous(),
                column=column.contiguous())
    return result


def lookup_1d(x: torch.Tensor, grid: torch.Tensor,
              values: torch.Tensor) -> torch.Tensor:
    """Clamped piecewise-linear interpolation with arbitrary batch shape."""

    x = x.clamp(grid[0], grid[-1])
    upper = torch.searchsorted(grid, x.contiguous(), right=True).clamp(
        1, grid.numel() - 1)
    lower = upper - 1
    x0, x1 = grid[lower], grid[upper]
    weight = (x - x0) / (x1 - x0).clamp_min(1.0e-12)
    return torch.lerp(values[lower], values[upper], weight)


def lookup_2d(row_value: torch.Tensor, column_value: torch.Tensor,
              row_grid: torch.Tensor, column_grid: torch.Tensor,
              values: torch.Tensor) -> torch.Tensor:
    """Clamped bilinear interpolation with arbitrary batch shape."""

    row_value = row_value.clamp(row_grid[0], row_grid[-1])
    column_value = column_value.clamp(column_grid[0], column_grid[-1])
    row_upper = torch.searchsorted(
        row_grid, row_value.contiguous(), right=True).clamp(
            1, row_grid.numel() - 1)
    column_upper = torch.searchsorted(
        column_grid, column_value.contiguous(), right=True).clamp(
            1, column_grid.numel() - 1)
    row_lower, column_lower = row_upper - 1, column_upper - 1
    row_weight = ((row_value - row_grid[row_lower]) /
                  (row_grid[row_upper] - row_grid[row_lower]).clamp_min(1e-12))
    column_weight = ((column_value - column_grid[column_lower]) /
                     (column_grid[column_upper] -
                      column_grid[column_lower]).clamp_min(1e-12))
    lower = torch.lerp(
        values[row_lower, column_lower], values[row_lower, column_upper],
        column_weight)
    upper = torch.lerp(
        values[row_upper, column_lower], values[row_upper, column_upper],
        column_weight)
    return torch.lerp(lower, upper, row_weight)
