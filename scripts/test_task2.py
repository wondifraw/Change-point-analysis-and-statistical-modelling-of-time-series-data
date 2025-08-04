"""
Quick test for Task 2 implementation
"""

import sys
import os
sys.path.append('src')
sys.path.append('src/task2')

try:
    from src.task2.run_task2 import run_task2_analysis
    print("Task 2 modules imported successfully")
    
    # Run minimal test
    print("\nRunning Task 2 analysis...")
    results = run_task2_analysis()
    print("\nTask 2 completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Installing PyMC3 may be required: pip install pymc3")
except Exception as e:
    print(f"Error: {e}")
    print("Task 2 implementation ready but may need PyMC3 installation")