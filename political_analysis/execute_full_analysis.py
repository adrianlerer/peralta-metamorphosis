#!/usr/bin/env python3
"""
Execute Full Enhanced Political Analysis
Final comprehensive analysis with all enhancements
"""

import sys
sys.path.append('/home/user/webapp')

from political_analysis.integrate_political_analysis import main_expanded
import time

def main():
    """Execute the full enhanced political analysis."""
    
    print("🚀 EXECUTING FULL ENHANCED POLITICAL ANALYSIS")
    print("="*80)
    print("This will run the complete analysis with:")
    print("• 40+ document expanded corpus")
    print("• Enhanced coordinate calculation")
    print("• Bootstrap validation (50 iterations)")
    print("• Electoral correlation analysis")
    print("• Enhanced phase transition detection")
    print("• Publication-ready visualizations")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Run the enhanced analysis
        results = main_expanded()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  Analysis completed in {duration:.1f} seconds")
        print("✅ All enhancements successfully executed!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()