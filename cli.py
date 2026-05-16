"""User input helpers for the CLI."""

from datetime import datetime


def get_float(prompt, min_val=None, max_val=None, allow_zero=False):
    while True:
        try:
            val = float(input(prompt))
            if not allow_zero and val == 0:
                print("  ⚠  Value cannot be zero.")
                continue
            if min_val is not None and val < min_val:
                print(f"  ⚠  Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  ⚠  Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("  ⚠  Please enter a valid number.")


def get_float_or_default(prompt: str, default: float, min_val: float, max_val: float) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if val < min_val:
                print(f"  ⚠  Must be >= {min_val}")
                continue
            if val > max_val:
                print(f"  ⚠  Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("  ⚠  Please enter a valid number or press Enter for default.")


def get_date(prompt):
    while True:
        try:
            s = input(prompt).strip()
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            print("  ⚠  Use format YYYY-MM-DD  (e.g. 2027-06-17)")


def get_option_type():
    while True:
        t = input("  Option type [C/P]: ").strip().upper()
        if t in ("C", "P"):
            return t
        print("  ⚠  Enter C for Call or P for Put.")


def get_int_choice(prompt: str, valid_range: range, default: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if val in valid_range:
                return val
            print(f"  ⚠  Must be between {valid_range.start} and {valid_range.stop - 1}")
        except ValueError:
            print("  ⚠  Please enter a valid integer.")
