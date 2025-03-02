import pandas as pd
from sdv.single_table import CTGANSynthesizer
import matplotlib.pyplot as plt

# Replace with your actual file path
file_path = r'C:\Users\Marcin\Desktop\healthcare_dataset.csv'

# Load the CSV file into a DataFrame
data = pd.read_csv(file_path)

#drop name colums
real = data.drop(['Name', 'Doctor', 'Hospital'], axis=1) 

real = real.head(1000) 

metadata = {
    "primary_key": None,  # No explicit primary key in the dataset
    "columns": {
        #"Name": {"sdtype": "text"},
        "Age": {"sdtype": "numerical", "computer_representation": "Float"},
        "Gender": {"sdtype": "categorical"},
        "Blood Type": {"sdtype": "categorical"},
        "Medical Condition": {"sdtype": "categorical"},
        "Date of Admission": {"sdtype": "datetime", "datetime_format": "%Y-%m-%d"},
        #"Doctor": {"sdtype": "text"},
        #"Hospital": {"sdtype": "categorical"},
        "Insurance Provider": {"sdtype": "categorical"},
        "Billing Amount": {"sdtype": "numerical", "computer_representation": "Float"},
        "Room Number": {"sdtype": "numerical", "computer_representation": "Float"},
        "Admission Type": {"sdtype": "categorical"},
        "Discharge Date": {"sdtype": "datetime", "datetime_format": "%Y-%m-%d"},
        "Medication": {"sdtype": "categorical"},
        "Test Results": {"sdtype": "categorical"}
    }
}


from sdv.metadata import SingleTableMetadata

# Convert metadata dictionary to SingleTableMetadata object
metadata_obj = SingleTableMetadata.load_from_dict(metadata)

# Initialize the synthesizer with the corrected metadata object
synthesizer = CTGANSynthesizer(metadata_obj)

# Fit the synthesizer
synthesizer.fit(real)

# Generate synthetic data
syn = synthesizer.sample(num_rows=len(real))

# View the generated data
print(syn.head())
