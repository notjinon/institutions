# Git Version Control Setup

## ✅ DONE - Your Scripts Are Now Under Version Control!

Your scripts have been committed to git. Here's what's tracked:

### 📝 Tracked (in version control):
- **All Python scripts** (.py files)
- **Configuration files** (.gitignore, data_sources.json)
- **Documentation** (README.md, FIXES_IMPLEMENTED.md)
- **Small reference data** (primarydates/*.csv)

### 🚫 NOT Tracked (excluded via .gitignore):
- **DIME data/** - Large CSVs and parquet files (2.5 GB each!)
- **outputs/** - Generated analysis files
- **__pycache__/** - Python cache files

## 🎯 Your Current Setup (Based on What I See):

1. **Large data files live in Box** (2-3 GB parquet files)
2. **Download them locally** using `python download_data.py`
3. **Local files go into:** `DIME data/YYYY_parquet/` folders
4. **Git ignores these** - they're too big for GitHub (max 100 MB per file)

## 🚀 Next Steps to Push to GitHub:

### 1. Create GitHub repo:
```bash
# Go to github.com and create a new repository
# Then connect it:
git remote add origin https://github.com/YOUR_USERNAME/institutions.git
git branch -M main
git push -u origin main
```

### 2. Daily workflow:
```bash
# After editing scripts:
git add .
git commit -m "Description of changes"
git push

# To see what changed:
git status
git diff
```

## 📊 Data Strategy (What You're Already Doing):

### Scripts (Git) ✅
- Version controlled on GitHub
- Small, text-based
- Easy to track changes

### Large Data Files (Box) ✅
- CSVs: ~100s of MB
- Parquet: ~2.5 GB each
- Stored in Box cloud
- Downloaded locally when needed
- Ignored by git (.gitignore)

### Reference Data (Git) ✅
- Small CSVs like primary dates
- Tracked in git
- Committed with scripts

## 🔄 Team Collaboration Setup:

```bash
# Teammate clones repo:
git clone https://github.com/YOUR_USERNAME/institutions.git
cd institutions

# They download the data:
python download_data.py --all

# Now they have:
# ✅ Scripts (from git)
# ✅ Data files (from Box via download script)
```

## ⚡ Quick Commands:

```bash
# Check what changed
git status

# Commit script changes
git add *.py
git commit -m "Updated matching logic"
git push

# Get latest changes
git pull

# See history
git log --oneline

# Create a branch for experiments
git checkout -b experiment-new-feature
```

## 🎁 Pro Tips:

1. **Never commit large data files** - GitHub limit is 100 MB per file
2. **Commit small, commit often** - Every script change deserves a commit
3. **Write meaningful commit messages** - Future you will thank you
4. **Keep data_sources.json updated** - So others can download the same data

## 🆘 If You Accidentally Add Large Files:

```bash
# Unstage a file:
git restore --staged DIME\ data/2000_candidate_donor.csv

# Remove from git history (if already committed):
git rm --cached "DIME data/*.csv"
```

---

**You're all set!** Your scripts are version controlled. Just create a GitHub repo and push! 🎉
