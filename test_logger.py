# test_logger.py
import pytest
import logging
import os

def test_logging():
    # Set up log directory and file with absolute path
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Reset any existing handlers
    )
    
    # Write log entry and flush
    logger = logging.getLogger()
    logger.info("Test log entry")
    for handler in logger.handlers:
        handler.flush()  # Ensure log is written to file
    
    # Verify file exists
    print(f"Checking if file exists: {log_file}")
    assert os.path.exists(log_file), f"Log file {log_file} was not created"
    
    # Verify content
    with open(log_file, 'r') as f:
        content = f.read()
        print(f"Log file content: {content}")
        assert "Test log entry" in content, "Log entry not found in file"
    
    # Clean up: Close handlers to release file
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)