#!/usr/bin/env python3
"""
Quick test of enhanced political analysis
"""

import sys
sys.path.append('/home/user/webapp')

from political_analysis.integrate_political_analysis import IntegratedPoliticalAnalysis

def test_enhanced_analysis():
    """Test the enhanced analysis with minimal bootstrap iterations."""
    
    print("🧪 Testing Enhanced Political Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = IntegratedPoliticalAnalysis()
    
    # Test corpus loading
    print("1. Testing expanded corpus loading...")
    try:
        documents_df = analyzer.load_expanded_political_documents()
        print(f"   ✅ Loaded {len(documents_df)} documents")
        
        # Check for [0,0,0,0] anomalies
        zero_positions = 0
        for pos in documents_df['political_position']:
            if pos == [0, 0, 0, 0] or all(x == 0 for x in pos):
                zero_positions += 1
        
        print(f"   📊 Zero positions found: {zero_positions}/{len(documents_df)}")
        
        # Show sample coordinates
        print("   📍 Sample coordinates:")
        for i, (_, doc) in enumerate(documents_df.head(3).iterrows()):
            author = doc['author']
            coords = doc['political_position']
            print(f"      {author}: {[f'{c:.2f}' for c in coords]}")
            
    except Exception as e:
        print(f"   ❌ Error loading corpus: {e}")
        return
    
    # Test enhanced coordinate calculation
    print("\n2. Testing coordinate calculation...")
    try:
        from political_analysis.expanded_political_corpus import calculate_enhanced_political_coordinates
        
        test_cases = [
            ("La patria es del pueblo, no de la oligarquía", "Juan Perón", "Discurso", 1945),
            ("El estado debe ser mínimo y eficiente", "Javier Milei", "Conferencia", 2023),
            ("Las provincias tienen derechos inalienables", "Justo José de Urquiza", "Proclama", 1852)
        ]
        
        for text, author, title, year in test_cases:
            coords = calculate_enhanced_political_coordinates(text, title, author, year)
            print(f"   {author}: {[f'{c:.2f}' for c in coords]}")
            
        print("   ✅ Coordinate calculation working")
        
    except Exception as e:
        print(f"   ❌ Error in coordinate calculation: {e}")
        return
    
    # Test phase transition detection
    print("\n3. Testing phase transition detection...")
    try:
        transitions = analyzer.detect_enhanced_phase_transitions(documents_df)
        print(f"   ✅ Detected {len(transitions)} phase transitions")
        
        if transitions:
            print("   🔄 Top transitions:")
            for i, trans in enumerate(transitions[:3]):
                year = trans['year']
                magnitude = trans['magnitude']
                print(f"      {i+1}. {year}: magnitude {magnitude:.2f}")
                
    except Exception as e:
        print(f"   ❌ Error in phase transition detection: {e}")
        return
    
    print("\n✅ Enhanced analysis components working correctly!")
    print("="*50)

if __name__ == "__main__":
    test_enhanced_analysis()