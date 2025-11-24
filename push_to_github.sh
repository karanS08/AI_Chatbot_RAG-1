#!/bin/bash

echo "=========================================="
echo "Push to New GitHub Repository"
echo "=========================================="
echo ""
echo "Follow these steps:"
echo ""
echo "1. First, create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name your repository (e.g., 'AI_Chatbot_Sugarcane_RAG')"
echo "   - Choose public or private"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo "   - Click 'Create repository'"
echo ""
echo "2. Copy the repository URL (it will look like:"
echo "   https://github.com/shashanktamaskar/YOUR_REPO_NAME.git)"
echo ""
read -p "Enter your new GitHub repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "Error: Repository URL cannot be empty"
    exit 1
fi

echo ""
echo "=========================================="
echo "Initializing Git and Pushing..."
echo "=========================================="
echo ""

# Check if .git exists
if [ -d ".git" ]; then
    echo "Git repository already exists."
    read -p "Do you want to add a new remote? (y/n): " ADD_REMOTE
    if [ "$ADD_REMOTE" = "y" ]; then
        git remote add new-origin "$REPO_URL"
        echo "Added new remote 'new-origin'"
        echo ""
        echo "Pushing to new repository..."
        git push new-origin main
    else
        echo "Checking current remote..."
        git remote -v
        echo ""
        read -p "Push to existing remote? (y/n): " PUSH_EXISTING
        if [ "$PUSH_EXISTING" = "y" ]; then
            git push origin main
        fi
    fi
else
    echo "Initializing new Git repository..."
    git init
    git add .
    git commit -m "Initial commit: AI Chatbot with RAG API endpoint"
    git branch -M main
    git remote add origin "$REPO_URL"
    echo ""
    echo "Pushing to GitHub..."
    git push -u origin main
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "Your repository should now be available at:"
echo "${REPO_URL%.git}"
echo ""
