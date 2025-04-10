# Makefile for Rice Anomaly Detection UI Project

# Variables
VENV_DIR = venv
PYTHON = python3
PIP = $(VENV_DIR)/bin/pip
APP = app.py  # Replace with the name of your main script file

# Default target
.PHONY: all
all: setup install run

# Create virtual environment
.PHONY: setup
setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created."

# Install dependencies
.PHONY: install
install: setup
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

# Run the Gradio app
.PHONY: run
run:
	@echo "Running the Gradio app..."
	$(VENV_DIR)/bin/python $(APP)

# Clean up virtual environment and generated files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleaned up."

# Rebuild everything from scratch
.PHONY: rebuild
rebuild: clean all

# Help message
.PHONY: help
help:
	@echo "Makefile for Rice Anomaly Detection UI Project"
	@echo ""
	@echo "Targets:"
	@echo "  all       : Set up environment, install dependencies, and run the app (default)"
	@echo "  setup     : Create a virtual environment"
	@echo "  install   : Install dependencies from requirements.txt"
	@echo "  run       : Run the Gradio app"
	@echo "  clean     : Remove virtual environment and generated files"
	@echo "  rebuild   : Clean and rebuild everything"
	@echo "  help      : Show this help message"