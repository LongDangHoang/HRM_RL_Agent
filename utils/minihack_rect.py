from dataclasses import dataclass


@dataclass
class MiniHackMapRect:
    first_row: int = 0
    last_row: int = 21
    first_col: int = 0
    last_col: int = 79


def get_rect(width: int, height: int):
    default_rect = MiniHackMapRect()

    w = default_rect.last_col + 1
    h = default_rect.last_row + 1

    y = round((h - height) / 2) - 1
    x = round((w - width) / 2) + 4
    cropped_rect = MiniHackMapRect(
        first_row=y,
        last_row=y + height,
        first_col=x,
        last_col=x + width,
    )
    return cropped_rect
