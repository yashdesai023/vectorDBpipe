# 🚀 Release & Deployment Guide for vectorDBpipe

This guide covers the final steps to **Push your code to GitHub** and **Publish your package to PyPI**.

---

## ✅ Prerequisites

Ensure you are in the project root directory:
```bash
cd e:\Private\AI-PROJECTS-PORTFOLIO-DOCS-ASSETS\ALL-PROJECTS-PACKAGES\vectorDBpipe
```

---

## 📦 Part 1: Publish to PyPI (The Python Package Index)

This makes your package installable via `pip install vectordbpipe`.

### Step 1: Install Twine
Twine is the tool used to securely upload packages to PyPI.
```bash
pip install twine
```

### Step 2: Upload Artifacts
We have already built the distribution files in the `dist/` folder. Use this command to upload them:

```bash
twine upload dist/*
```

### Step 3: Enter Credentials
When prompted:
*   **Username**: `__token__`  (Literally type `__token__`)
*   **Password**: Paste your **PyPI API Token** (starts with `pypi-...`).
    *   *Note: usage of username/password is deprecated; tokens are required.*

---

## 🐙 Part 2: Push to GitHub

This saves your source code and makes the repository public/private.

### Step 1: Initialize Git (If not already done)
```bash
git init
git branch -M main
```

### Step 2: Add Files & Commit
```bash
git add .
git commit -m "Release v0.1.3: Production Ready with Pinecone V3 & Batch Processing"
```

### Step 3: Tag the Release (Best Practice)
Creates a formal release point in history.
```bash
git tag v0.1.3
```

### Step 4: Push to Remote
Replace `YOUR_GITHUB_URL` with your actual repository URL (e.g., `https://github.com/yashdesai023/vectorDBpipe.git`).

```bash
git remote add origin https://github.com/yashdesai023/vectorDBpipe.git
git push -u origin main --tags
```

---

## 🧪 Part 3: Verification

Once uploaded, wait ~2 minutes, then verify it works by installing it in a fresh environment:

```bash
# In a new terminal window
pip install vectordbpipe
python -c "import vectorDBpipe; print(vectorDBpipe.__version__)"
```

**🎉 Success! You are live.**
