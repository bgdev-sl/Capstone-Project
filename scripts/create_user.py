# Create/update local user account

from __future__ import annotations
import argparse
from getpass import getpass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ids.auth import create_user_store  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", choices=["ADMIN", "ANALYST"], required=True)
    parser.add_argument("--output", default="configs/users.json")
    args = parser.parse_args()

    password = getpass("Password: ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")

    create_user_store(ROOT / args.output, args.username, password, args.role)
    print(f"Created/updated {args.role} user '{args.username}'.")

if __name__ == "__main__":
    main()