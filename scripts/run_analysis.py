#!/usr/bin/env python3
"""
Automated analysis runner script
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_workflow import DataAnalysisWorkflow
from event_compiler import EventCompiler

def main():
    """Run complete analysis pipeline"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Compile events
        compiler = EventCompiler("data/processed/events.csv")
        events_df = compiler.compile_major_events()
        print(f"✓ Compiled {len(events_df)} events")
        
        # Run workflow
        workflow = DataAnalysisWorkflow("data/raw/brent_oil_prices.csv")
        results = workflow.execute_workflow()
        print("✓ Analysis completed successfully")
        
        return 0
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())