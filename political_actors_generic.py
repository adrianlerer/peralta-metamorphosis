"""
Political Actor Network Dataset - Generic Framework
Political Similarity Framework (PSF) for Multi-Dimensional Analysis
Date: September 12, 2025

This dataset uses generic reference actors for political similarity analysis
without specific individual references.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_generic_political_dataset():
    """
    Create political dataset with generic reference actors
    """
    
    actors_data = [
        # REFERENCE ACTORS FOR SIMILARITY ANALYSIS
        {
            'name': 'Actor Histórico A',
            'period': '1970-1975',
            'country': 'Argentina',
            'era': 'Historical',
            'position': 'Government Official',
            'ideology_economic': 0.2,
            'ideology_social': 0.8,
            'leadership_messianic': 0.9,
            'leadership_charismatic': 0.7,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.95,
            'populist_appeal': 0.7,
            'authoritarian': 0.85,
            'media_savvy': 0.3,
            'violence_associated': 0.9,
            'notes': 'Historical reference with high mystical elements'
        },
        {
            'name': 'Actor Contemporáneo B',
            'period': '2020-present',
            'country': 'Argentina',
            'era': 'Contemporary',
            'position': 'Political Leader',
            'ideology_economic': 0.95,
            'ideology_social': 0.6,
            'leadership_messianic': 0.85,
            'leadership_charismatic': 0.9,
            'anti_establishment': 0.95,
            'symbolic_mystical': 0.8,
            'populist_appeal': 0.9,
            'authoritarian': 0.7,
            'media_savvy': 0.95,
            'violence_associated': 0.2,
            'notes': 'Contemporary reference with high media presence'
        },
        {
            'name': 'Juan Domingo Perón',
            'period': '1946-1955, 1973-1974',
            'country': 'Argentina',
            'era': 'Historical',
            'position': 'President',
            'ideology_economic': 0.3,
            'ideology_social': 0.4,
            'leadership_messianic': 0.6,
            'leadership_charismatic': 0.95,
            'anti_establishment': 0.7,
            'symbolic_mystical': 0.3,
            'populist_appeal': 0.9,
            'authoritarian': 0.7,
            'media_savvy': 0.8,
            'violence_associated': 0.4,
            'notes': 'Classic populist leader, mass appeal'
        },
        {
            'name': 'Eva Perón',
            'period': '1946-1952',
            'country': 'Argentina',
            'era': 'Historical',
            'position': 'First Lady',
            'ideology_economic': 0.2,
            'ideology_social': 0.3,
            'leadership_messianic': 0.7,
            'leadership_charismatic': 0.9,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.4,
            'populist_appeal': 0.95,
            'authoritarian': 0.3,
            'media_savvy': 0.8,
            'violence_associated': 0.1,
            'notes': 'Spiritual connection with working class'
        },
        {
            'name': 'Carlos Menem',
            'period': '1989-1999',
            'country': 'Argentina',
            'era': 'Historical',
            'position': 'President',
            'ideology_economic': 0.8,
            'ideology_social': 0.5,
            'leadership_messianic': 0.4,
            'leadership_charismatic': 0.7,
            'anti_establishment': 0.3,
            'symbolic_mystical': 0.2,
            'populist_appeal': 0.6,
            'authoritarian': 0.5,
            'media_savvy': 0.8,
            'violence_associated': 0.3,
            'notes': 'Neoliberal populist, showman president'
        },
        
        # ADDITIONAL POLITICAL ACTORS
        {
            'name': 'Hugo Chávez',
            'period': '1999-2013',
            'country': 'Venezuela',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.1,
            'ideology_social': 0.2,
            'leadership_messianic': 0.85,
            'leadership_charismatic': 0.9,
            'anti_establishment': 0.9,
            'symbolic_mystical': 0.6,
            'populist_appeal': 0.9,
            'authoritarian': 0.8,
            'media_savvy': 0.9,
            'violence_associated': 0.6,
            'notes': 'Bolivarian mysticism, media presence'
        },
        {
            'name': 'Donald Trump',
            'period': '2017-2021',
            'country': 'United States',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.6,
            'ideology_social': 0.7,
            'leadership_messianic': 0.7,
            'leadership_charismatic': 0.8,
            'anti_establishment': 0.85,
            'symbolic_mystical': 0.3,
            'populist_appeal': 0.8,
            'authoritarian': 0.7,
            'media_savvy': 0.9,
            'violence_associated': 0.6,
            'notes': 'Media-driven populist leadership'
        },
        {
            'name': 'Viktor Orbán',
            'period': '1998-2002, 2010-present',
            'country': 'Hungary',
            'era': 'Contemporary',
            'position': 'Prime Minister',
            'ideology_economic': 0.6,
            'ideology_social': 0.9,
            'leadership_messianic': 0.6,
            'leadership_charismatic': 0.7,
            'anti_establishment': 0.7,
            'symbolic_mystical': 0.4,
            'populist_appeal': 0.7,
            'authoritarian': 0.85,
            'media_savvy': 0.8,
            'violence_associated': 0.3,
            'notes': 'Illiberal democracy, nationalist populism'
        },
        {
            'name': 'Cristina Fernández de Kirchner',
            'period': '2007-2015',
            'country': 'Argentina',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.2,
            'ideology_social': 0.3,
            'leadership_messianic': 0.4,
            'leadership_charismatic': 0.7,
            'anti_establishment': 0.6,
            'symbolic_mystical': 0.2,
            'populist_appeal': 0.7,
            'authoritarian': 0.5,
            'media_savvy': 0.6,
            'violence_associated': 0.2,
            'notes': 'Peronist continuation, institutional conflicts'
        },
        {
            'name': 'Mauricio Macri',
            'period': '2015-2019',
            'country': 'Argentina',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.8,
            'ideology_social': 0.6,
            'leadership_messianic': 0.2,
            'leadership_charismatic': 0.4,
            'anti_establishment': 0.3,
            'symbolic_mystical': 0.1,
            'populist_appeal': 0.4,
            'authoritarian': 0.3,
            'media_savvy': 0.5,
            'violence_associated': 0.1,
            'notes': 'Business-oriented, technocratic approach'
        },
        {
            'name': 'Alberto Fernández',
            'period': '2019-2023',
            'country': 'Argentina',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.3,
            'ideology_social': 0.4,
            'leadership_messianic': 0.2,
            'leadership_charismatic': 0.3,
            'anti_establishment': 0.3,
            'symbolic_mystical': 0.1,
            'populist_appeal': 0.4,
            'authoritarian': 0.3,
            'media_savvy': 0.4,
            'violence_associated': 0.1,
            'notes': 'Academic background, moderate approach'
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(actors_data)
    
    # Calculate composite scores
    df['messianic_total'] = (df['leadership_messianic'] + df['symbolic_mystical']) / 2
    df['populist_total'] = (df['populist_appeal'] + df['anti_establishment']) / 2
    df['charisma_total'] = (df['leadership_charismatic'] + df['media_savvy']) / 2
    
    # Calculate Political Similarity Index (PSI) for each actor
    reference_profile = df[df['name'] == 'Actor Histórico A'].iloc[0]
    
    similarity_dimensions = [
        'ideology_economic', 'ideology_social', 'leadership_messianic', 
        'leadership_charismatic', 'anti_establishment', 'symbolic_mystical',
        'populist_appeal', 'authoritarian', 'media_savvy'
    ]
    
    def calculate_political_similarity(row, reference_row, dimensions):
        """Calculate Political Similarity Index (PSI)"""
        distances = []
        for dim in dimensions:
            dist = abs(row[dim] - reference_row[dim])
            distances.append(1 - dist)  # Convert distance to similarity
        return np.mean(distances)
    
    df['political_similarity_index'] = df.apply(
        lambda row: calculate_political_similarity(row, reference_profile, similarity_dimensions), 
        axis=1
    )
    
    # Add metadata
    df['dataset_version'] = '3.0_generic'
    df['creation_date'] = datetime.now().isoformat()
    df['total_actors'] = len(df)
    df['framework'] = 'Political Similarity Framework (PSF)'
    
    return df

def get_political_similarity_breakdown():
    """
    Get detailed breakdown of Political Similarity Framework analysis
    """
    df = create_generic_political_dataset()
    
    actor_a = df[df['name'] == 'Actor Histórico A'].iloc[0]
    actor_b = df[df['name'] == 'Actor Contemporáneo B'].iloc[0]
    
    dimensions = {
        'ideological_alignment': {
            'economic': abs(actor_a['ideology_economic'] - actor_b['ideology_economic']),
            'social': abs(actor_a['ideology_social'] - actor_b['ideology_social'])
        },
        'leadership_style': {
            'messianic': abs(actor_a['leadership_messianic'] - actor_b['leadership_messianic']),
            'charismatic': abs(actor_a['leadership_charismatic'] - actor_b['leadership_charismatic'])
        },
        'anti_establishment_rhetoric': {
            'anti_establishment': abs(actor_a['anti_establishment'] - actor_b['anti_establishment']),
            'populist_appeal': abs(actor_a['populist_appeal'] - actor_b['populist_appeal'])
        },
        'symbolic_elements': {
            'symbolic_mystical': abs(actor_a['symbolic_mystical'] - actor_b['symbolic_mystical']),
            'media_savvy': abs(actor_a['media_savvy'] - actor_b['media_savvy'])
        }
    }
    
    # Convert distances to similarities and calculate dimension scores
    breakdown = {}
    for category, dims in dimensions.items():
        similarities = {k: 1 - v for k, v in dims.items()}
        breakdown[category] = {
            'individual_scores': similarities,
            'category_mean': np.mean(list(similarities.values()))
        }
    
    overall_similarity = np.mean([cat['category_mean'] for cat in breakdown.values()])
    
    return {
        'breakdown': breakdown,
        'overall_similarity': overall_similarity,
        'actor_a_profile': actor_a.to_dict(),
        'actor_b_profile': actor_b.to_dict()
    }

if __name__ == "__main__":
    # Create and display dataset
    df = create_generic_political_dataset()
    print(f"Created Political Similarity Framework dataset with {len(df)} actors")
    print(f"Columns: {list(df.columns)}")
    
    # Show similarity analysis
    breakdown = get_political_similarity_breakdown()
    print(f"\nPolitical Similarity Index Analysis: {breakdown['overall_similarity']:.3f}")
    
    for category, data in breakdown['breakdown'].items():
        print(f"\n{category.replace('_', ' ').title()}: {data['category_mean']:.3f}")
        for dim, score in data['individual_scores'].items():
            print(f"  - {dim}: {score:.3f}")