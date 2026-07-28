# Updating the GitHub repository

These files are intended to be copied into the existing local repository.

## Files to add or replace

```text
trajectory_dataset_core.py
generate_ou_range_dataset.py
generate_three_model_dataset.py
README.md
requirements.txt
.gitignore
CHANGELOG.md
```

## Commit and push

From Windows Command Prompt, inside the local repository folder:

```bat
git status
git pull --rebase origin main
git add trajectory_dataset_core.py generate_ou_range_dataset.py generate_three_model_dataset.py README.md requirements.txt .gitignore CHANGELOG.md
git commit -m "Add OU-range and three-model dataset generators"
git push origin main
```

If the remote is not configured yet, check it with:

```bat
git remote -v
```

Then add it, replacing `REPOSITORY-NAME` with the actual repository name:

```bat
git remote add origin https://github.com/nicoforti/REPOSITORY-NAME.git
git branch -M main
git push -u origin main
```

## Optional version tag

After pushing the commit:

```bat
git tag -a v2.0.0 -m "OU-range and OU/CT three-model dataset generators"
git push origin v2.0.0
```

A GitHub Release can then be created from tag `v2.0.0` through the repository's Releases page.
