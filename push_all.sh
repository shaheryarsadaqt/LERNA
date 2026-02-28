#!/bin/bash
echo "=========================================="
echo "🚀 Pushing LERNA to ALL remotes"
echo "=========================================="

# GitHub
echo "📤 Pushing to GitHub (origin)..."
git push origin main
git push origin baseline-runs
git push origin lerna-switching

# GitLab Group (sheheryarsadaqat)
echo "📤 Pushing to GitLab (sheheryarsadaqat-group)..."
git push gitlab main
git push gitlab baseline-runs
git push gitlab lerna-switching

echo "=========================================="
echo "✅ Done! Pushed to both remotes"
echo "=========================================="
