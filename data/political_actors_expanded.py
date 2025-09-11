"""
Extended Political Actor Network Dataset for Multi-Dimensional Analysis
Paper 11: Expanded Network Analysis with 30+ Actors
Date: September 11, 2025

This dataset expands the political actor network to 30+ figures across different eras,
ideologies, and leadership styles for comprehensive multi-dimensional analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_expanded_political_dataset():
    """
    Create expanded dataset with 30+ political actors across multiple dimensions:
    - Ideological alignment
    - Leadership style (messianic/charismatic)
    - Anti-establishment rhetoric
    - Symbolic/mystical elements
    - Populist characteristics
    - Authoritarian tendencies
    """
    
    actors_data = [
        # HISTORICAL ARGENTINE FIGURES
        {
            'name': 'José López Rega',
            'period': '1973-1975',
            'country': 'Argentina',
            'era': 'Historical',
            'position': 'Minister of Social Welfare',
            'ideology_economic': 0.2,  # Center-left economics
            'ideology_social': 0.8,    # Conservative social
            'leadership_messianic': 0.9,    # Highly messianic
            'leadership_charismatic': 0.7,  # Moderately charismatic
            'anti_establishment': 0.8,      # Strong anti-establishment
            'symbolic_mystical': 0.95,      # Extreme mystical elements
            'populist_appeal': 0.7,
            'authoritarian': 0.85,
            'media_savvy': 0.3,
            'violence_associated': 0.9,
            'notes': 'AAA founder, astrology, esoteric practices'
        },
        {
            'name': 'Javier Milei',
            'period': '2021-present',
            'country': 'Argentina',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.95,  # Far-right economics
            'ideology_social': 0.6,     # Conservative social
            'leadership_messianic': 0.85,   # Highly messianic
            'leadership_charismatic': 0.9,  # Highly charismatic
            'anti_establishment': 0.95,     # Extreme anti-establishment
            'symbolic_mystical': 0.8,       # Strong symbolic elements
            'populist_appeal': 0.9,
            'authoritarian': 0.7,
            'media_savvy': 0.95,
            'violence_associated': 0.2,
            'notes': 'Tantric sex, spiritual elements, chainsaw symbolism'
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
            'notes': 'Spiritual connection with descamisados'
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
        
        # REGIONAL LATIN AMERICAN FIGURES
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
            'notes': 'Bolivarian mysticism, Sunday TV shows'
        },
        {
            'name': 'Evo Morales',
            'period': '2006-2019',
            'country': 'Bolivia',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.2,
            'ideology_social': 0.1,
            'leadership_messianic': 0.7,
            'leadership_charismatic': 0.8,
            'anti_establishment': 0.85,
            'symbolic_mystical': 0.8,
            'populist_appeal': 0.85,
            'authoritarian': 0.6,
            'media_savvy': 0.6,
            'violence_associated': 0.4,
            'notes': 'Indigenous spirituality, Pachamama ceremonies'
        },
        {
            'name': 'Jair Bolsonaro',
            'period': '2019-2022',
            'country': 'Brazil',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.7,
            'ideology_social': 0.9,
            'leadership_messianic': 0.6,
            'leadership_charismatic': 0.5,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.4,
            'populist_appeal': 0.7,
            'authoritarian': 0.8,
            'media_savvy': 0.7,
            'violence_associated': 0.7,
            'notes': 'Military nostalgia, evangelical connections'
        },
        {
            'name': 'Rafael Correa',
            'period': '2007-2017',
            'country': 'Ecuador',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.3,
            'ideology_social': 0.4,
            'leadership_messianic': 0.5,
            'leadership_charismatic': 0.7,
            'anti_establishment': 0.7,
            'symbolic_mystical': 0.3,
            'populist_appeal': 0.75,
            'authoritarian': 0.6,
            'media_savvy': 0.8,
            'violence_associated': 0.3,
            'notes': 'Academic populist, media control'
        },
        {
            'name': 'Daniel Ortega',
            'period': '1985-1990, 2007-present',
            'country': 'Nicaragua',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.2,
            'ideology_social': 0.3,
            'leadership_messianic': 0.6,
            'leadership_charismatic': 0.6,
            'anti_establishment': 0.6,
            'symbolic_mystical': 0.4,
            'populist_appeal': 0.6,
            'authoritarian': 0.9,
            'media_savvy': 0.5,
            'violence_associated': 0.8,
            'notes': 'Sandinista mystique, revolutionary symbolism'
        },
        
        # GLOBAL POPULIST FIGURES
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
            'notes': 'Twitter presidency, rally messianism'
        },
        {
            'name': 'Marine Le Pen',
            'period': '2011-present',
            'country': 'France',
            'era': 'Contemporary',
            'position': 'Party Leader',
            'ideology_economic': 0.5,
            'ideology_social': 0.8,
            'leadership_messianic': 0.4,
            'leadership_charismatic': 0.6,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.2,
            'populist_appeal': 0.75,
            'authoritarian': 0.6,
            'media_savvy': 0.7,
            'violence_associated': 0.3,
            'notes': 'National Rally, anti-immigration'
        },
        {
            'name': 'Matteo Salvini',
            'period': '2013-present',
            'country': 'Italy',
            'era': 'Contemporary',
            'position': 'Party Leader',
            'ideology_economic': 0.4,
            'ideology_social': 0.8,
            'leadership_messianic': 0.5,
            'leadership_charismatic': 0.7,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.3,
            'populist_appeal': 0.8,
            'authoritarian': 0.6,
            'media_savvy': 0.8,
            'violence_associated': 0.4,
            'notes': 'Lega Nord, social media populism'
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
            'notes': 'Illiberal democracy, Christian nationalism'
        },
        {
            'name': 'Geert Wilders',
            'period': '2006-present',
            'country': 'Netherlands',
            'era': 'Contemporary',
            'position': 'Party Leader',
            'ideology_economic': 0.5,
            'ideology_social': 0.85,
            'leadership_messianic': 0.3,
            'leadership_charismatic': 0.5,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.1,
            'populist_appeal': 0.7,
            'authoritarian': 0.5,
            'media_savvy': 0.7,
            'violence_associated': 0.2,
            'notes': 'Anti-Islam rhetoric, blonde iconography'
        },
        
        # HISTORICAL COMPARISON FIGURES
        {
            'name': 'Adolf Hitler',
            'period': '1933-1945',
            'country': 'Germany',
            'era': 'Historical',
            'position': 'Führer',
            'ideology_economic': 0.5,
            'ideology_social': 0.95,
            'leadership_messianic': 0.95,
            'leadership_charismatic': 0.9,
            'anti_establishment': 0.9,
            'symbolic_mystical': 0.8,
            'populist_appeal': 0.8,
            'authoritarian': 0.99,
            'media_savvy': 0.8,
            'violence_associated': 0.99,
            'notes': 'Occult interests, Thule Society connections'
        },
        {
            'name': 'Benito Mussolini',
            'period': '1922-1943',
            'country': 'Italy',
            'era': 'Historical',
            'position': 'Duce',
            'ideology_economic': 0.6,
            'ideology_social': 0.8,
            'leadership_messianic': 0.8,
            'leadership_charismatic': 0.85,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.4,
            'populist_appeal': 0.75,
            'authoritarian': 0.9,
            'media_savvy': 0.7,
            'violence_associated': 0.8,
            'notes': 'Theatrical leadership, Roman symbolism'
        },
        {
            'name': 'Francisco Franco',
            'period': '1939-1975',
            'country': 'Spain',
            'era': 'Historical',
            'position': 'Caudillo',
            'ideology_economic': 0.7,
            'ideology_social': 0.9,
            'leadership_messianic': 0.6,
            'leadership_charismatic': 0.4,
            'anti_establishment': 0.5,
            'symbolic_mystical': 0.5,
            'populist_appeal': 0.4,
            'authoritarian': 0.95,
            'media_savvy': 0.4,
            'violence_associated': 0.9,
            'notes': 'Catholic mysticism, traditionalist'
        },
        
        # MYSTICAL/ESOTERIC POLITICAL FIGURES
        {
            'name': 'Rasputin',
            'period': '1905-1916',
            'country': 'Russia',
            'era': 'Historical',
            'position': 'Court Advisor',
            'ideology_economic': 0.3,
            'ideology_social': 0.7,
            'leadership_messianic': 0.95,
            'leadership_charismatic': 0.9,
            'anti_establishment': 0.4,
            'symbolic_mystical': 0.99,
            'populist_appeal': 0.3,
            'authoritarian': 0.3,
            'media_savvy': 0.2,
            'violence_associated': 0.2,
            'notes': 'Pure mystical influence, healing powers'
        },
        {
            'name': 'Julius Evola',
            'period': '1920-1970s',
            'country': 'Italy',
            'era': 'Historical',
            'position': 'Philosopher',
            'ideology_economic': 0.8,
            'ideology_social': 0.95,
            'leadership_messianic': 0.7,
            'leadership_charismatic': 0.6,
            'anti_establishment': 0.9,
            'symbolic_mystical': 0.95,
            'populist_appeal': 0.2,
            'authoritarian': 0.9,
            'media_savvy': 0.3,
            'violence_associated': 0.4,
            'notes': 'Traditionalist esotericism, influenced far-right'
        },
        
        # TECHNOCRATIC/RATIONAL LEADERS (CONTRAST GROUP)
        {
            'name': 'Angela Merkel',
            'period': '2005-2021',
            'country': 'Germany',
            'era': 'Contemporary',
            'position': 'Chancellor',
            'ideology_economic': 0.6,
            'ideology_social': 0.5,
            'leadership_messianic': 0.1,
            'leadership_charismatic': 0.3,
            'anti_establishment': 0.1,
            'symbolic_mystical': 0.05,
            'populist_appeal': 0.2,
            'authoritarian': 0.2,
            'media_savvy': 0.6,
            'violence_associated': 0.05,
            'notes': 'Technocratic, rational leadership'
        },
        {
            'name': 'Justin Trudeau',
            'period': '2015-present',
            'country': 'Canada',
            'era': 'Contemporary',
            'position': 'Prime Minister',
            'ideology_economic': 0.4,
            'ideology_social': 0.2,
            'leadership_messianic': 0.2,
            'leadership_charismatic': 0.7,
            'anti_establishment': 0.2,
            'symbolic_mystical': 0.1,
            'populist_appeal': 0.5,
            'authoritarian': 0.2,
            'media_savvy': 0.8,
            'violence_associated': 0.05,
            'notes': 'Centrist charisma, social media savvy'
        },
        {
            'name': 'Emmanuel Macron',
            'period': '2017-present',
            'country': 'France',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.7,
            'ideology_social': 0.3,
            'leadership_messianic': 0.3,
            'leadership_charismatic': 0.6,
            'anti_establishment': 0.4,
            'symbolic_mystical': 0.1,
            'populist_appeal': 0.3,
            'authoritarian': 0.3,
            'media_savvy': 0.7,
            'violence_associated': 0.1,
            'notes': 'Technocratic populism, Jupiter complex'
        },
        
        # ADDITIONAL ARGENTINE FIGURES
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
            'notes': 'Peronist continuation, media battles'
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
            'notes': 'Business-oriented, rational approach'
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
            'notes': 'Academic background, moderate Peronist'
        },
        
        # ADDITIONAL MYSTICAL LEADERS
        {
            'name': 'Ayatollah Khomeini',
            'period': '1979-1989',
            'country': 'Iran',
            'era': 'Historical',
            'position': 'Supreme Leader',
            'ideology_economic': 0.3,
            'ideology_social': 0.95,
            'leadership_messianic': 0.9,
            'leadership_charismatic': 0.8,
            'anti_establishment': 0.9,
            'symbolic_mystical': 0.9,
            'populist_appeal': 0.8,
            'authoritarian': 0.9,
            'media_savvy': 0.5,
            'violence_associated': 0.7,
            'notes': 'Religious mysticism, revolutionary Shia Islam'
        },
        {
            'name': 'Narendra Modi',
            'period': '2014-present',
            'country': 'India',
            'era': 'Contemporary',
            'position': 'Prime Minister',
            'ideology_economic': 0.6,
            'ideology_social': 0.8,
            'leadership_messianic': 0.6,
            'leadership_charismatic': 0.8,
            'anti_establishment': 0.5,
            'symbolic_mystical': 0.6,
            'populist_appeal': 0.8,
            'authoritarian': 0.6,
            'media_savvy': 0.8,
            'violence_associated': 0.4,
            'notes': 'Hindu nationalism, RSS background'
        },
        {
            'name': 'Recep Tayyip Erdoğan',
            'period': '2003-present',
            'country': 'Turkey',
            'era': 'Contemporary',
            'position': 'President',
            'ideology_economic': 0.5,
            'ideology_social': 0.8,
            'leadership_messianic': 0.6,
            'leadership_charismatic': 0.8,
            'anti_establishment': 0.7,
            'symbolic_mystical': 0.4,
            'populist_appeal': 0.8,
            'authoritarian': 0.8,
            'media_savvy': 0.7,
            'violence_associated': 0.5,
            'notes': 'Islamic democracy, Ottoman nostalgia'
        },
        
        # FINAL ADDITIONS TO REACH 30+
        {
            'name': 'Beppe Grillo',
            'period': '2009-present',
            'country': 'Italy',
            'era': 'Contemporary',
            'position': 'Party Founder',
            'ideology_economic': 0.3,
            'ideology_social': 0.4,
            'leadership_messianic': 0.4,
            'leadership_charismatic': 0.8,
            'anti_establishment': 0.9,
            'symbolic_mystical': 0.2,
            'populist_appeal': 0.85,
            'authoritarian': 0.3,
            'media_savvy': 0.9,
            'violence_associated': 0.1,
            'notes': 'Comedian turned politician, Five Star Movement'
        },
        {
            'name': 'Pablo Iglesias',
            'period': '2014-2021',
            'country': 'Spain',
            'era': 'Contemporary',
            'position': 'Party Leader',
            'ideology_economic': 0.1,
            'ideology_social': 0.2,
            'leadership_messianic': 0.3,
            'leadership_charismatic': 0.7,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.1,
            'populist_appeal': 0.8,
            'authoritarian': 0.2,
            'media_savvy': 0.8,
            'violence_associated': 0.1,
            'notes': 'Podemos founder, left-wing populism'
        },
        {
            'name': 'Alexis Tsipras',
            'period': '2015-2019',
            'country': 'Greece',
            'era': 'Contemporary',
            'position': 'Prime Minister',
            'ideology_economic': 0.1,
            'ideology_social': 0.3,
            'leadership_messianic': 0.3,
            'leadership_charismatic': 0.6,
            'anti_establishment': 0.8,
            'symbolic_mystical': 0.1,
            'populist_appeal': 0.8,
            'authoritarian': 0.2,
            'media_savvy': 0.6,
            'violence_associated': 0.1,
            'notes': 'Syriza leader, anti-austerity populism'
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(actors_data)
    
    # Calculate composite scores
    df['messianic_total'] = (df['leadership_messianic'] + df['symbolic_mystical']) / 2
    df['populist_total'] = (df['populist_appeal'] + df['anti_establishment']) / 2
    df['charisma_total'] = (df['leadership_charismatic'] + df['media_savvy']) / 2
    
    # Calculate López Rega similarity for each actor
    lopez_rega_profile = df[df['name'] == 'José López Rega'].iloc[0]
    
    similarity_dimensions = [
        'ideology_economic', 'ideology_social', 'leadership_messianic', 
        'leadership_charismatic', 'anti_establishment', 'symbolic_mystical',
        'populist_appeal', 'authoritarian', 'media_savvy'
    ]
    
    def calculate_similarity(row, reference_row, dimensions):
        """Calculate multi-dimensional similarity score"""
        distances = []
        for dim in dimensions:
            dist = abs(row[dim] - reference_row[dim])
            distances.append(1 - dist)  # Convert distance to similarity
        return np.mean(distances)
    
    df['lopez_rega_similarity'] = df.apply(
        lambda row: calculate_similarity(row, lopez_rega_profile, similarity_dimensions), 
        axis=1
    )
    
    # Add metadata
    df['dataset_version'] = '2.0_expanded'
    df['creation_date'] = datetime.now().isoformat()
    df['total_actors'] = len(df)
    
    return df

def get_multidimensional_breakdown():
    """
    Get detailed breakdown of López Rega-Milei similarity by dimensions
    """
    df = create_expanded_political_dataset()
    
    lopez_rega = df[df['name'] == 'José López Rega'].iloc[0]
    milei = df[df['name'] == 'Javier Milei'].iloc[0]
    
    dimensions = {
        'ideological_alignment': {
            'economic': abs(lopez_rega['ideology_economic'] - milei['ideology_economic']),
            'social': abs(lopez_rega['ideology_social'] - milei['ideology_social'])
        },
        'leadership_style': {
            'messianic': abs(lopez_rega['leadership_messianic'] - milei['leadership_messianic']),
            'charismatic': abs(lopez_rega['leadership_charismatic'] - milei['leadership_charismatic'])
        },
        'anti_establishment_rhetoric': {
            'anti_establishment': abs(lopez_rega['anti_establishment'] - milei['anti_establishment']),
            'populist_appeal': abs(lopez_rega['populist_appeal'] - milei['populist_appeal'])
        },
        'symbolic_mystical_elements': {
            'symbolic_mystical': abs(lopez_rega['symbolic_mystical'] - milei['symbolic_mystical']),
            'media_savvy': abs(lopez_rega['media_savvy'] - milei['media_savvy'])
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
        'lopez_rega_profile': lopez_rega.to_dict(),
        'milei_profile': milei.to_dict()
    }

if __name__ == "__main__":
    # Create and save dataset
    df = create_expanded_political_dataset()
    print(f"Created expanded dataset with {len(df)} political actors")
    print(f"Columns: {list(df.columns)}")
    
    # Show López Rega-Milei similarity breakdown
    breakdown = get_multidimensional_breakdown()
    print(f"\nLópez Rega-Milei Overall Similarity: {breakdown['overall_similarity']:.3f}")
    
    for category, data in breakdown['breakdown'].items():
        print(f"\n{category.replace('_', ' ').title()}: {data['category_mean']:.3f}")
        for dim, score in data['individual_scores'].items():
            print(f"  - {dim}: {score:.3f}")