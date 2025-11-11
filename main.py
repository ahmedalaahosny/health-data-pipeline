import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthDataPipeline:
    """
    ETL Data Pipeline for Healthcare Systems
    Processes medical data from EHR systems, performs data cleaning,
    transformation, and loads into databases.
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.validation_report = {}
        self.processing_stats = {}
        
    def extract_ehr_data(self, n_records=1000):
        """
        Extract synthetic EHR data from healthcare systems.
        """
        logger.info(f"Extracting {n_records} patient records...")
        
        np.random.seed(42)
        
        # Generate synthetic patient data
        patient_ids = [f"PAT{str(i).zfill(6)}" for i in range(1, n_records + 1)]
        ages = np.random.randint(18, 85, n_records)
        genders = np.random.choice(['M', 'F'], n_records)
        visit_dates = [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
        
        # Medical data
        blood_pressure_systolic = np.random.randint(90, 200, n_records)
        blood_pressure_diastolic = np.random.randint(60, 120, n_records)
        heart_rate = np.random.randint(50, 100, n_records)
        glucose_level = np.random.randint(70, 300, n_records)
        cholesterol = np.random.randint(100, 400, n_records)
        
        # Diagnoses and procedures
        diagnoses = np.random.choice(['Hypertension', 'Diabetes', 'COPD', 'Heart Disease', 'Anxiety'], n_records)
        procedures = np.random.choice(['Blood Test', 'X-Ray', 'ECG', 'Ultrasound', 'CT Scan'], n_records)
        
        self.raw_data = pd.DataFrame({
            'patient_id': patient_ids,
            'age': ages,
            'gender': genders,
            'visit_date': visit_dates,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'heart_rate': heart_rate,
            'glucose_level': glucose_level,
            'cholesterol': cholesterol,
            'diagnosis': diagnoses,
            'procedure': procedures
        })
        
        logger.info(f"Successfully extracted {len(self.raw_data)} records")
        return self.raw_data
    
    def validate_data(self) -> Dict:
        """
        Validate extracted data for quality and completeness.
        """
        logger.info("Validating data...")
        
        validation_results = {
            'total_records': len(self.raw_data),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'duplicate_records': self.raw_data.duplicated().sum(),
            'valid_records': len(self.raw_data) - self.raw_data.duplicated().sum(),
            'validation_score': 0
        }
        
        # Calculate validation score
        total_cells = len(self.raw_data) * len(self.raw_data.columns)
        missing_cells = self.raw_data.isnull().sum().sum()
        validation_results['validation_score'] = ((total_cells - missing_cells) / total_cells) * 100
        
        self.validation_report = validation_results
        logger.info(f"Validation Score: {validation_results['validation_score']:.2f}%")
        
        return validation_results
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and standardize healthcare data.
        """
        logger.info("Cleaning data...")
        
        cleaned_data = self.raw_data.copy()
        
        # Handle missing values
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype == 'object':
                cleaned_data[col].fillna('Unknown', inplace=True)
            else:
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
        
        # Remove duplicates
        cleaned_data.drop_duplicates(subset=['patient_id', 'visit_date'], inplace=True)
        
        # Standardize data types
        cleaned_data['visit_date'] = pd.to_datetime(cleaned_data['visit_date'])
        cleaned_data['age'] = cleaned_data['age'].astype(int)
        cleaned_data['heart_rate'] = cleaned_data['heart_rate'].astype(int)
        
        # Validate numeric ranges
        cleaned_data['blood_pressure_systolic'] = cleaned_data['blood_pressure_systolic'].clip(70, 250)
        cleaned_data['blood_pressure_diastolic'] = cleaned_data['blood_pressure_diastolic'].clip(40, 150)
        cleaned_data['glucose_level'] = cleaned_data['glucose_level'].clip(40, 400)
        cleaned_data['cholesterol'] = cleaned_data['cholesterol'].clip(50, 500)
        
        logger.info(f"Cleaned data: {len(cleaned_data)} records")
        return cleaned_data
    
    def transform_data(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform and enrich data for analytics.
        """
        logger.info("Transforming data...")
        
        transformed_data = cleaned_data.copy()
        
        # Create feature flags
        transformed_data['is_hypertensive'] = (transformed_data['blood_pressure_systolic'] > 140).astype(int)
        transformed_data['is_hyperglycemic'] = (transformed_data['glucose_level'] > 126).astype(int)
        transformed_data['high_cholesterol'] = (transformed_data['cholesterol'] > 200).astype(int)
        
        # Risk categorization
        def calculate_risk_score(row):
            score = 0
            if row['age'] > 60: score += 2
            if row['is_hypertensive']: score += 3
            if row['is_hyperglycemic']: score += 3
            if row['high_cholesterol']: score += 2
            return min(score, 10)
        
        transformed_data['risk_score'] = transformed_data.apply(calculate_risk_score, axis=1)
        transformed_data['risk_category'] = pd.cut(transformed_data['risk_score'], 
                                                    bins=[0, 3, 6, 10],
                                                    labels=['Low', 'Medium', 'High'])
        
        # Add processing timestamp
        transformed_data['processing_date'] = datetime.now()
        
        # Create patient hash for privacy
        transformed_data['patient_hash'] = transformed_data['patient_id'].apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )
        
        self.processed_data = transformed_data
        logger.info(f"Transformed data: {len(transformed_data)} records")
        
        return transformed_data
    
    def generate_report(self) -> Dict:
        """
        Generate pipeline processing report with visualizations.
        """
        logger.info("Generating report...")
        
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Age distribution
        axes[0, 0].hist(self.processed_data['age'], bins=20, color='#3498db', edgecolor='black')
        axes[0, 0].set_title('Patient Age Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Count')
        
        # Blood pressure distribution
        axes[0, 1].scatter(self.processed_data['blood_pressure_systolic'], 
                          self.processed_data['blood_pressure_diastolic'],
                          alpha=0.5, c=self.processed_data['risk_score'], cmap='YlOrRd')
        axes[0, 1].set_title('Blood Pressure Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Systolic')
        axes[0, 1].set_ylabel('Diastolic')
        
        # Risk category distribution
        risk_counts = self.processed_data['risk_category'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        axes[1, 0].bar(risk_counts.index, risk_counts.values, color=colors)
        axes[1, 0].set_title('Patient Risk Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Patients')
        
        # Processing statistics
        stats_text = f"""Pipeline Statistics:
- Total Records Processed: {len(self.processed_data):,}
- Validation Score: {self.validation_report.get('validation_score', 0):.2f}%
- Duplicate Records Removed: {self.validation_report.get('duplicate_records', 0)}
- High-Risk Patients: {(self.processed_data['risk_category'] == 'High').sum()}
- Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('pipeline_report.png', dpi=100, bbox_inches='tight')
        
        return {
            'total_records': len(self.processed_data),
            'validation_score': self.validation_report.get('validation_score', 0),
            'duplicates_removed': self.validation_report.get('duplicate_records', 0),
            'high_risk_count': int((self.processed_data['risk_category'] == 'High').sum()),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def run_pipeline(self):
        """
        Execute complete ETL pipeline.
        """
        logger.info("Starting ETL Pipeline...")
        logger.info("="*50)
        
        # Extract
        self.extract_ehr_data(n_records=1000)
        
        # Validate
        validation = self.validate_data()
        print(f"\nValidation Results:")
        print(f"Total Records: {validation['total_records']}")
        print(f"Validation Score: {validation['validation_score']:.2f}%")
        
        # Clean
        cleaned = self.clean_data()
        
        # Transform
        transformed = self.transform_data(cleaned)
        
        # Generate Report
        report = self.generate_report()
        print(f"\nPipeline Execution Report:")
        for key, value in report.items():
            print(f"{key}: {value}")
        
        logger.info("="*50)
        logger.info("ETL Pipeline Completed Successfully")
        
        return report

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = HealthDataPipeline()
    results = pipeline.run_pipeline()
    print(f"\nReport saved as pipeline_report.png")
