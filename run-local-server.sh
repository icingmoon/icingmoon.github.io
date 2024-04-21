#!/bin/bash

# Directory where Jekyll generates the site
JEKYLL_DEST_DIR="_site"

# Function to clean up generated files
cleanup() {
    echo "Cleaning up..."
    # Add command to clean up generated files
    # This example removes the default Jekyll site generation directory
    rm -rf "$JEKYLL_DEST_DIR"
    echo "Cleanup done."
}

# Trap signals and errors (EXIT, INT, TERM) to ensure cleanup is performed
trap cleanup EXIT INT TERM

# Function to build the Jekyll site
build_site() {
    echo "Building Jekyll site..."
    if ! bundle exec jekyll build; then
        echo "Error building site, exiting."
        exit 1
    fi
}

# Function to serve the site locally
serve_site() {
    echo "Serving Jekyll site locally..."
    # Serve the site in the background so we can trap signals
    bundle exec jekyll serve &
    # Capture the background process's PID
    JEKYLL_PID=$!
    # Wait for the Jekyll server process to finish
    wait $JEKYLL_PID
}

# Main script execution starts here

# Build the site, script exits if this fails
build_site

# Serve the site locally
serve_site

# Script will clean up generated files upon exit due to the trap set earlier
