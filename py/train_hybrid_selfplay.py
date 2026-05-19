from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tribes_rl.train import main as package_main

    package_main()


if __name__ == "__main__":
    main()
