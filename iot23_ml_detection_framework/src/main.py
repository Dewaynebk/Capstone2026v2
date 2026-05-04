"""Convenience entry point.

Running `python src/main.py` performs prediction using the processed IoT-23 data.
If preprocessing or model training has not been run yet, the program creates what it needs.
"""
from predict import main

if __name__ == "__main__":
    main()
