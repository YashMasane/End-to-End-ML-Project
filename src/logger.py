import logging
import os
from datetime import datetime

# Define the log file with a timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Specify the logs directory
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)  # Only create the logs directory

# Complete log file path
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure logging to use LOG_FILE_PATH
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Corrected filename argument
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

