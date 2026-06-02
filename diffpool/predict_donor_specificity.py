import argparse
import os

# ============================== args input ==============================
def validate_path(value):
    """Verification structure path: cannot be empty, and the path must exist"""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("Structure path cannot be empty")
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(f"Structure path does not exist: {value}")
    return os.path.abspath(value)


def validate_fold_type(value):
    """Verification fold type: can only be GTA or GTB"""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("Fold type cannot be empty")
    value = value.strip().upper()
    if value not in ("GTA", "GTB"):
        raise argparse.ArgumentTypeError(f"Fold type can only be GTA or GTB, current input: {value}")
    return value


def validate_output(value):
    """Verification output file name: cannot be empty, and the directory must be writable (if it exists)"""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("Output file name cannot be empty")
    # Check if the file name contains invalid characters (Windows / Linux compatible)
    invalid_chars = '<>:"/\\|?*'
    basename = os.path.basename(value)
    if any(c in basename for c in invalid_chars):
        raise argparse.ArgumentTypeError(f"Output file name contains invalid characters: {basename}")
    # Check if the target directory is writable (if the directory exists)
    dirname = os.path.dirname(value) or "."
    if os.path.exists(dirname) and not os.access(dirname, os.W_OK):
        raise argparse.ArgumentTypeError(f"Output directory is not writable: {dirname}")
    return os.path.abspath(value)

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction of donor specificity for glycosyltransferases")
    parser.add_argument(
        "-p", "--path", required=True, type=validate_path, help="Structure path, must exist (required)", metavar="PATH"
    )
    parser.add_argument(
        "-t", "--type", required=True, type=validate_fold_type, help="Fold type, optional values: GTA / GTB (required)", metavar="TYPE"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=validate_output, help="Output file name (required)", metavar="FILE"
    )
    return parser.parse_args()


args = parse_args()
print(f"Structure path:   {args.path}")
print(f"Fold type:   {args.type}")
print(f"Output file:   {args.output}")
























