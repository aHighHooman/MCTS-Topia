from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

from .belief import BELIEF_PLANE_NAMES
from .encoding import normalize_message


BOARD_MATRIX_KEYS = ("terrain", "resource", "building", "city", "unit", "exp", "road")
POSITION_KEYS = ("destination", "position", "target_pos")


def transform_xy(x: Any, y: Any, size: int, rotation: int = 0, mirror: bool = False) -> tuple[int, int]:
    new_x = int(x)
    new_y = int(y)
    max_coord = int(size) - 1
    if mirror:
        new_x = max_coord - new_x
    for _ in range(int(rotation) % 4):
        new_x, new_y = max_coord - new_y, new_x
    return new_x, new_y


def transform_message_symmetry(
    message: Mapping[str, Any],
    *,
    rotation: int = 0,
    mirror: bool = False,
) -> Dict[str, Any]:
    normalized = normalize_message(copy.deepcopy(dict(message)))
    board = normalized.get("observation", {}).get("board", {})
    size = int(board.get("size", 0) or 0)
    if size <= 0:
        return normalized
    if rotation % 2 and _board_height(board) not in (0, size):
        return normalized

    _transform_board(board, size, rotation, mirror)
    for unit in normalized.get("observation", {}).get("units", []) or []:
        _transform_object_position(unit, size, rotation, mirror)
    for city in normalized.get("observation", {}).get("cities", []) or []:
        _transform_object_position(city, size, rotation, mirror)
        for building in city.get("buildings", []) or []:
            if isinstance(building, dict):
                _transform_object_position(building, size, rotation, mirror)
    _transform_lighthouses(board.get("lighthouses"), size, rotation, mirror)
    _transform_belief(normalized.get("observation", {}).get("belief"), size, rotation, mirror)
    for action in normalized.get("actions", []) or []:
        _transform_action(action, size, rotation, mirror)
    return normalized


def _board_height(board: Mapping[str, Any]) -> int:
    tiles = board.get("tiles")
    if isinstance(tiles, list) and tiles:
        return len(tiles)
    for key in BOARD_MATRIX_KEYS:
        matrix = board.get(key)
        if isinstance(matrix, list) and matrix:
            return len(matrix)
    return 0


def _transform_board(board: Dict[str, Any], size: int, rotation: int, mirror: bool) -> None:
    for key in BOARD_MATRIX_KEYS:
        matrix = board.get(key)
        if isinstance(matrix, list):
            board[key] = _transform_matrix(matrix, size, rotation, mirror)
    tiles = board.get("tiles")
    if isinstance(tiles, list):
        board["tiles"] = _transform_tiles(tiles, size, rotation, mirror)


def _transform_matrix(matrix: list[Any], size: int, rotation: int, mirror: bool) -> list[list[Any]]:
    out = [[None for _ in range(size)] for _ in range(size)]
    for y, row in enumerate(matrix[:size]):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row[:size]):
            new_x, new_y = transform_xy(x, y, size, rotation, mirror)
            out[new_y][new_x] = copy.deepcopy(value)
    return out


def _transform_tiles(tiles: list[Any], size: int, rotation: int, mirror: bool) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any] | None]] = [[None for _ in range(size)] for _ in range(size)]
    for y, row in enumerate(tiles[:size]):
        if not isinstance(row, list):
            continue
        for x, tile in enumerate(row[:size]):
            if not isinstance(tile, dict):
                continue
            new_x, new_y = transform_xy(tile.get("x", x), tile.get("y", y), size, rotation, mirror)
            transformed = copy.deepcopy(tile)
            transformed["x"] = new_x
            transformed["y"] = new_y
            out[new_y][new_x] = transformed
    return [[cell if cell is not None else {"x": x, "y": y} for x, cell in enumerate(row)] for y, row in enumerate(out)]


def _transform_object_position(item: Dict[str, Any], size: int, rotation: int, mirror: bool) -> None:
    if "x" not in item or "y" not in item:
        return
    item["x"], item["y"] = transform_xy(item["x"], item["y"], size, rotation, mirror)


def _transform_lighthouses(lighthouses: Any, size: int, rotation: int, mirror: bool) -> None:
    if not isinstance(lighthouses, dict):
        return
    for value in lighthouses.values():
        if isinstance(value, dict):
            _transform_object_position(value, size, rotation, mirror)


def _transform_belief(belief: Any, size: int, rotation: int, mirror: bool) -> None:
    if not isinstance(belief, dict):
        return
    planes = belief.get("planes")
    if not isinstance(planes, dict):
        return
    for name in BELIEF_PLANE_NAMES:
        matrix = planes.get(name)
        if isinstance(matrix, list):
            planes[name] = _transform_matrix(matrix, size, rotation, mirror)


def _transform_action(action: Dict[str, Any], size: int, rotation: int, mirror: bool) -> None:
    _transform_object_position(action, size, rotation, mirror)
    for key in POSITION_KEYS:
        payload = action.get(key)
        if isinstance(payload, dict):
            _transform_object_position(payload, size, rotation, mirror)
