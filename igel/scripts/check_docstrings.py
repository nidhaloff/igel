import subprocess
import sys

def main():
    # You can change 'igel' to the directory you want to check
    result = subprocess.run(
        ["pydocstyle", "igel"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("All docstrings are present and correct! 🎉")
        sys.exit(0)
    else:
        print("Missing or incorrect docstrings found:\n")
        print(result.stdout)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
