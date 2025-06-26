import logging
import os
from datetime import datetime

# Generate a timestamped log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path for logs
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the directory for logs if it does not exist
os.makedirs(logs_path, exist_ok=True)

# Set the final log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Log the path for confirmation
logging.info(f"Logging initialized, log file created at: {LOG_FILE_PATH}")
