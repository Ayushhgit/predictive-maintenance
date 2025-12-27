from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.constants import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

data_transformation = DataTransformation()
data_validation = DataValidation()
model_trainer = ModelTrainer()

# Run validation on your CSV file
raw_csv_path = RAW_DATA_DIR
validated_path = data_validation.initiate_data_validation(raw_csv_path)

print(f"✓ Validation completed!")
print(f"✓ Validated file saved at: {validated_path}")


print("Step 3: Running data transformation...")
X_train_path, X_test_path, y_train_path, y_test_path = (
    data_transformation.initiate_data_transformation(validated_data_path=validated_path)
)

print(f"✓ Transformation completed!")
print(f"✓ Transformed files saved in the directory: {TRANSFORMED_DATA_DIR}")
print(f"  - X_train: {X_train_path}")
print(f"  - X_test: {X_test_path}")
print(f"  - y_train: {y_train_path}")
print(f"  - y_test: {y_test_path}")

## Model Trainer
print(f"Running Model Trainer")
results = model_trainer.initiate_model_training(
    X_train_path=X_train_path,
    X_test_path=X_test_path,
    y_train_path=y_train_path,
    y_test_path=y_test_path,
)
print(f"Model Training completed")
print(results)
