from src.components.data_ingestion import DataIngestion


def main():
    CONFIG_PATH = "config/config.yaml"
    SCHEMA_PATH = "config/schema.yaml"

    ingestion = DataIngestion(config_path=CONFIG_PATH, schema_path=SCHEMA_PATH)
    ingestion.run()


if __name__ == "__main__":
    main()
