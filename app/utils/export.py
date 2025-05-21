"""
Export utility for saving detection results to CSV files.
"""
import os
import csv
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultExporter:
    """Export detection results to CSV files."""
    
    def __init__(self, export_dir: str = None):
        """
        Initialize the ResultExporter.
        
        Args:
            export_dir: Directory to save exported files (defaults to 'exports' in current directory)
        """
        self.export_dir = export_dir or os.path.join(os.getcwd(), "exports")
        os.makedirs(self.export_dir, exist_ok=True)
        
        # Define CSV headers
        self.headers = [
            "timestamp",
            "input_text",
            "classification",
            "action",
            "justification",
            "detailed_reasoning",
            "policy_references"
        ]
    
    def export_to_csv(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export a single result to a CSV file.
        
        Args:
            result: The detection result to export
            filename: Optional filename (defaults to timestamp-based name)
            
        Returns:
            Path to the exported file
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detection_result_{timestamp}.csv"
            
            # Ensure it has .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
                
            filepath = os.path.join(self.export_dir, filename)
            
            # Format policy references for CSV
            policy_refs = "; ".join([f"{doc.get('source', 'Unknown')}" for doc in result.get("policy_references", [])])
            
            # Prepare row data
            row_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_text": result.get("input_text", ""),
                "classification": result.get("classification", ""),
                "action": result.get("recommended_action", ""),
                "justification": result.get("action_justification", ""),
                "detailed_reasoning": result.get("detailed_reasoning", ""),
                "policy_references": policy_refs
            }
            
            # Write to CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                writer.writeheader()
                writer.writerow(row_data)
                
            logger.info(f"Result exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting result to CSV: {e}")
            raise
    
    def export_multiple_to_csv(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Export multiple results to a single CSV file.
        
        Args:
            results: List of detection results to export
            filename: Optional filename (defaults to timestamp-based name)
            
        Returns:
            Path to the exported file
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detection_results_{timestamp}.csv"
            
            # Ensure it has .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
                
            filepath = os.path.join(self.export_dir, filename)
            
            # Write to CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                writer.writeheader()
                
                for result in results:
                    # Format policy references for CSV
                    policy_refs = "; ".join([f"{doc.get('source', 'Unknown')}" for doc in result.get("policy_references", [])])
                    
                    # Prepare row data
                    row_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "input_text": result.get("input_text", ""),
                        "classification": result.get("classification", ""),
                        "action": result.get("recommended_action", ""),
                        "justification": result.get("action_justification", ""),
                        "detailed_reasoning": result.get("detailed_reasoning", ""),
                        "policy_references": policy_refs
                    }
                    
                    writer.writerow(row_data)
                
            logger.info(f"Multiple results exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting multiple results to CSV: {e}")
            raise