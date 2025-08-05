from src.components.data_validation import DataValidation
from src.constants import RAW_DATA_DIR
data_validation = DataValidation()

# Run validation on your CSV file
raw_csv_path = RAW_DATA_DIR  
validated_path = data_validation.initiate_data_validation(raw_csv_path)

print(f"✓ Validation completed!")
print(f"✓ Validated file saved at: {validated_path}")