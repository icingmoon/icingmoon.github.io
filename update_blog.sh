#!/bin/bash

# Check if commit message was supplied
if [ -z "\$1" ]; then
  echo "Warning: Commit message is no supplied."
  echo "Usage: \$0 \"Your commit message\", default commit message is current date and time."
  COMMIT_MESSAGE="${1:-$(date +'%Y-%m-%d %H:%M:%S')}"
else
  # Assign the first argument as the commit message
  COMMIT_MESSAGE="\$1"
fi

# Add changes to the staging area
git add .

# Commit the changes
git commit -m "${COMMIT_MESSAGE}"

# Push the changes to the remote master branch
git push origin master

echo "Changes have been committed and pushed to the master branch."