"""
Bootstrap Statistical Validation Module for Paper 11
Implements bootstrap resampling for similarity analysis robustness testing
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BootstrapValidator:
    """
    Bootstrap validation class for political actor similarity analysis.
    Implements multiple bootstrap strategies for robustness testing.
    """
    
    def __init__(self, n_iterations: int = 1000, confidence_level: float = 0.95, 
                 random_state: Optional[int] = 42):
        """
        Initialize bootstrap validator.
        
        Parameters:
        -----------
        n_iterations : int
            Number of bootstrap iterations (default: 1000)
        confidence_level : float
            Confidence level for intervals (default: 0.95)
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Store results
        self.bootstrap_results = {}
        self.confidence_intervals = {}
        
    def bootstrap_similarity_matrix(self, data: pd.DataFrame, 
                                  similarity_function, 
                                  dimensions: List[str]) -> Dict:
        """
        Bootstrap validation of similarity matrix calculation.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Political actors data
        similarity_function : callable
            Function to calculate similarity matrix
        dimensions : list
            Political dimensions to analyze
            
        Returns:
        --------
        dict : Bootstrap results including confidence intervals
        """
        logger.info(f"Starting bootstrap validation with {self.n_iterations} iterations")
        
        n_actors = len(data)
        bootstrap_similarities = []
        
        # Original calculation
        original_matrix = similarity_function(data, dimensions)
        
        for i in range(self.n_iterations):
            if i % 100 == 0:
                logger.info(f"Bootstrap iteration {i}/{self.n_iterations}")
                
            # Bootstrap sample with replacement
            bootstrap_indices = np.random.choice(n_actors, size=n_actors, replace=True)
            bootstrap_data = data.iloc[bootstrap_indices].copy()
            
            try:
                # Calculate similarity matrix for bootstrap sample
                bootstrap_matrix = similarity_function(bootstrap_data, dimensions)
                bootstrap_similarities.append(bootstrap_matrix)
            except Exception as e:
                logger.warning(f"Error in bootstrap iteration {i}: {e}")
                continue
        
        # Calculate statistics
        bootstrap_array = np.array(bootstrap_similarities)
        
        results = {
            'original_matrix': original_matrix,
            'bootstrap_matrices': bootstrap_array,
            'mean_matrix': np.mean(bootstrap_array, axis=0),
            'std_matrix': np.std(bootstrap_array, axis=0),
            'n_successful_iterations': len(bootstrap_similarities)
        }
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        results['ci_lower'] = np.percentile(bootstrap_array, lower_percentile, axis=0)
        results['ci_upper'] = np.percentile(bootstrap_array, upper_percentile, axis=0)
        
        self.bootstrap_results['similarity_matrix'] = results
        return results
    
    def bootstrap_lopez_rega_milei_comparison(self, data: pd.DataFrame, 
                                            comparison_function) -> Dict:
        """
        Bootstrap validation specifically for López Rega-Milei similarity comparison.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Political actors data including López Rega and Milei
        comparison_function : callable
            Function to calculate López Rega-Milei similarity
            
        Returns:
        --------
        dict : Bootstrap results for the comparison
        """
        logger.info("Bootstrap validation for López Rega-Milei comparison")
        
        bootstrap_similarities = []
        
        # Original calculation
        original_similarity = comparison_function(data)
        
        for i in range(self.n_iterations):
            # Bootstrap sample
            n_actors = len(data)
            bootstrap_indices = np.random.choice(n_actors, size=n_actors, replace=True)
            bootstrap_data = data.iloc[bootstrap_indices].copy()
            
            try:
                # Ensure López Rega and Milei are in bootstrap sample
                lopez_rega_present = any('López Rega' in str(name) for name in bootstrap_data['actor'])
                milei_present = any('Milei' in str(name) for name in bootstrap_data['actor'])
                
                if not (lopez_rega_present and milei_present):
                    # Add them if missing
                    lopez_rega_data = data[data['actor'].str.contains('López Rega', na=False)]
                    milei_data = data[data['actor'].str.contains('Milei', na=False)]
                    
                    if not lopez_rega_data.empty and not milei_data.empty:
                        bootstrap_data = pd.concat([bootstrap_data, lopez_rega_data, milei_data])
                
                similarity = comparison_function(bootstrap_data)
                bootstrap_similarities.append(similarity)
                
            except Exception as e:
                logger.warning(f"Error in López Rega-Milei bootstrap iteration {i}: {e}")
                continue
        
        # Calculate statistics
        bootstrap_array = np.array(bootstrap_similarities)
        
        results = {
            'original_similarity': original_similarity,
            'bootstrap_similarities': bootstrap_array,
            'mean_similarity': np.mean(bootstrap_array),
            'std_similarity': np.std(bootstrap_array),
            'median_similarity': np.median(bootstrap_array),
            'n_successful_iterations': len(bootstrap_similarities)
        }
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        results['ci_lower'] = np.percentile(bootstrap_array, lower_percentile)
        results['ci_upper'] = np.percentile(bootstrap_array, upper_percentile)
        
        # Statistical tests
        results['p_value_greater_than_median'] = np.sum(bootstrap_array > np.median(bootstrap_array)) / len(bootstrap_array)
        
        self.bootstrap_results['lopez_rega_milei'] = results
        return results
    
    def bootstrap_multidimensional_analysis(self, data: pd.DataFrame, 
                                          analysis_function,
                                          dimension_categories: Dict[str, List[str]]) -> Dict:
        """
        Bootstrap validation for multidimensional analysis breakdown.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Political actors data
        analysis_function : callable
            Function to perform multidimensional analysis
        dimension_categories : dict
            Categories of political dimensions
            
        Returns:
        --------
        dict : Bootstrap results by dimension category
        """
        logger.info("Bootstrap validation for multidimensional analysis")
        
        results = {}
        
        for category, dimensions in dimension_categories.items():
            logger.info(f"Bootstrapping category: {category}")
            
            bootstrap_results = []
            
            # Original calculation
            original_result = analysis_function(data, dimensions)
            
            for i in range(self.n_iterations):
                # Bootstrap sample
                n_actors = len(data)
                bootstrap_indices = np.random.choice(n_actors, size=n_actors, replace=True)
                bootstrap_data = data.iloc[bootstrap_indices].copy()
                
                try:
                    result = analysis_function(bootstrap_data, dimensions)
                    bootstrap_results.append(result)
                except Exception as e:
                    logger.warning(f"Error in multidimensional bootstrap iteration {i}, category {category}: {e}")
                    continue
            
            # Calculate statistics
            if bootstrap_results:
                bootstrap_array = np.array(bootstrap_results)
                
                category_results = {
                    'original_result': original_result,
                    'bootstrap_results': bootstrap_array,
                    'mean_result': np.mean(bootstrap_array, axis=0) if bootstrap_array.ndim > 1 else np.mean(bootstrap_array),
                    'std_result': np.std(bootstrap_array, axis=0) if bootstrap_array.ndim > 1 else np.std(bootstrap_array),
                    'n_successful_iterations': len(bootstrap_results)
                }
                
                # Confidence intervals
                alpha = 1 - self.confidence_level
                lower_percentile = (alpha/2) * 100
                upper_percentile = (1 - alpha/2) * 100
                
                category_results['ci_lower'] = np.percentile(bootstrap_array, lower_percentile, axis=0)
                category_results['ci_upper'] = np.percentile(bootstrap_array, upper_percentile, axis=0)
                
                results[category] = category_results
        
        self.bootstrap_results['multidimensional'] = results
        return results
    
    def jackknife_validation(self, data: pd.DataFrame, 
                           analysis_function) -> Dict:
        """
        Jackknife validation (leave-one-out) for robustness testing.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Political actors data
        analysis_function : callable
            Function to analyze data
            
        Returns:
        --------
        dict : Jackknife validation results
        """
        logger.info("Performing jackknife validation")
        
        n_actors = len(data)
        jackknife_results = []
        
        # Original calculation
        original_result = analysis_function(data)
        
        for i in range(n_actors):
            # Leave one out
            jackknife_data = data.drop(data.index[i])
            
            try:
                result = analysis_function(jackknife_data)
                jackknife_results.append(result)
            except Exception as e:
                logger.warning(f"Error in jackknife iteration {i}: {e}")
                continue
        
        # Calculate statistics
        jackknife_array = np.array(jackknife_results)
        
        results = {
            'original_result': original_result,
            'jackknife_results': jackknife_array,
            'mean_result': np.mean(jackknife_array, axis=0) if jackknife_array.ndim > 1 else np.mean(jackknife_array),
            'std_result': np.std(jackknife_array, axis=0) if jackknife_array.ndim > 1 else np.std(jackknife_array),
            'bias': np.mean(jackknife_array, axis=0) - original_result if jackknife_array.ndim > 1 else np.mean(jackknife_array) - original_result,
            'n_successful_iterations': len(jackknife_results)
        }
        
        return results
    
    def sensitivity_analysis(self, data: pd.DataFrame, 
                           analysis_function,
                           parameter_ranges: Dict) -> Dict:
        """
        Sensitivity analysis by varying key parameters.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Political actors data
        analysis_function : callable
            Function to analyze data (should accept **kwargs)
        parameter_ranges : dict
            Dictionary of parameter names and their ranges to test
            
        Returns:
        --------
        dict : Sensitivity analysis results
        """
        logger.info("Performing sensitivity analysis")
        
        results = {}
        
        for param_name, param_values in parameter_ranges.items():
            logger.info(f"Testing sensitivity for parameter: {param_name}")
            
            param_results = []
            
            for param_value in param_values:
                try:
                    kwargs = {param_name: param_value}
                    result = analysis_function(data, **kwargs)
                    param_results.append({
                        'parameter_value': param_value,
                        'result': result
                    })
                except Exception as e:
                    logger.warning(f"Error in sensitivity analysis for {param_name}={param_value}: {e}")
                    continue
            
            results[param_name] = param_results
        
        return results
    
    def generate_bootstrap_report(self) -> str:
        """
        Generate comprehensive bootstrap validation report.
        
        Returns:
        --------
        str : Formatted report text
        """
        report = []
        report.append("=" * 60)
        report.append("BOOTSTRAP VALIDATION REPORT - PAPER 11")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Bootstrap Parameters:")
        report.append(f"- Iterations: {self.n_iterations}")
        report.append(f"- Confidence Level: {self.confidence_level}")
        report.append(f"- Random State: {self.random_state}")
        report.append("")
        
        # Similarity matrix results
        if 'similarity_matrix' in self.bootstrap_results:
            results = self.bootstrap_results['similarity_matrix']
            report.append("SIMILARITY MATRIX BOOTSTRAP RESULTS:")
            report.append("-" * 40)
            report.append(f"Successful iterations: {results['n_successful_iterations']}")
            report.append(f"Original matrix shape: {results['original_matrix'].shape}")
            report.append(f"Bootstrap mean similarity (overall): {np.mean(results['mean_matrix']):.4f}")
            report.append(f"Bootstrap std (overall): {np.mean(results['std_matrix']):.4f}")
            report.append("")
        
        # López Rega-Milei results
        if 'lopez_rega_milei' in self.bootstrap_results:
            results = self.bootstrap_results['lopez_rega_milei']
            report.append("LÓPEZ REGA-MILEI COMPARISON BOOTSTRAP RESULTS:")
            report.append("-" * 50)
            report.append(f"Original similarity: {results['original_similarity']:.4f}")
            report.append(f"Bootstrap mean: {results['mean_similarity']:.4f}")
            report.append(f"Bootstrap median: {results['median_similarity']:.4f}")
            report.append(f"Bootstrap std: {results['std_similarity']:.4f}")
            report.append(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
            report.append(f"Successful iterations: {results['n_successful_iterations']}")
            report.append("")
        
        # Multidimensional results
        if 'multidimensional' in self.bootstrap_results:
            report.append("MULTIDIMENSIONAL ANALYSIS BOOTSTRAP RESULTS:")
            report.append("-" * 45)
            for category, results in self.bootstrap_results['multidimensional'].items():
                report.append(f"\n{category.upper()}:")
                report.append(f"  Successful iterations: {results['n_successful_iterations']}")
                if isinstance(results['mean_result'], np.ndarray):
                    report.append(f"  Mean result shape: {results['mean_result'].shape}")
                else:
                    report.append(f"  Mean result: {results['mean_result']:.4f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """
        Save bootstrap results to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save results
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'bootstrap_results': self.bootstrap_results,
                'confidence_intervals': self.confidence_intervals,
                'parameters': {
                    'n_iterations': self.n_iterations,
                    'confidence_level': self.confidence_level,
                    'random_state': self.random_state
                }
            }, f)
        
        logger.info(f"Bootstrap results saved to {filepath}")

# Helper functions for common bootstrap operations
def calculate_bootstrap_ci(data: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Parameters:
    -----------
    data : np.ndarray
        Bootstrap samples
    confidence_level : float
        Confidence level
        
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    return np.percentile(data, lower_percentile), np.percentile(data, upper_percentile)

def bootstrap_hypothesis_test(observed_statistic: float, 
                            bootstrap_statistics: np.ndarray,
                            alternative: str = 'two-sided') -> float:
    """
    Perform bootstrap hypothesis test.
    
    Parameters:
    -----------
    observed_statistic : float
        Observed test statistic
    bootstrap_statistics : np.ndarray
        Bootstrap distribution of test statistic
    alternative : str
        Alternative hypothesis ('two-sided', 'greater', 'less')
        
    Returns:
    --------
    float : p-value
    """
    n_bootstrap = len(bootstrap_statistics)
    
    if alternative == 'two-sided':
        p_value = 2 * min(
            np.sum(bootstrap_statistics >= observed_statistic) / n_bootstrap,
            np.sum(bootstrap_statistics <= observed_statistic) / n_bootstrap
        )
    elif alternative == 'greater':
        p_value = np.sum(bootstrap_statistics >= observed_statistic) / n_bootstrap
    elif alternative == 'less':
        p_value = np.sum(bootstrap_statistics <= observed_statistic) / n_bootstrap
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    return min(p_value, 1.0)