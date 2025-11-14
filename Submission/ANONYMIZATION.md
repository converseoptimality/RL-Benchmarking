# Anonymization & Submission Guide

This repo is prepared for **double‑blind review**. Follow these steps before sharing the link:

## 1) Hygiene in the repo
- **Remove author IDs** from files (names, emails, institutions).
- **Scrub notebooks** to drop outputs & metadata:
  ```bash
  pip install nbstripout
  nbstripout --install  # adds a git filter
  git ls-files '*.ipynb' | xargs nbstripout
  ```
- **Check config & history** (optional):
  ```bash
  git config user.name "anonymous"
  git config user.email "anonymous@users.noreply.github.com"
  ```
  If your history contains names/URLs, consider a fresh repo or filtered history.

## 2) Anonymous GitHub (anonymous.4open.science)
1. Push this blinded repo to GitHub (can be public or private).
2. Visit **https://anonymous.4open.science** and log in with GitHub.
3. Select your repo, configure the **anonymization terms** (project names, lab, authors), and choose an **expiry**.
4. Generate the anonymous link and paste it in your paper / OpenReview as the code link.
5. The mirror auto‑updates at most **once per hour** after new commits. Large binaries or non‑text files are not mirrored.

**Notes / limitations**
- The service focuses on **text files**; very large or binary assets may not be served.
- Reviewers generally **cannot `git clone`** the anonymous mirror. If cloning is required, provide a zip artifact (see below).

## 3) Optional: zip artifact for reviewers
Create a minimal archive of the exact version you submit:
```bash
git archive --format=zip --output=artifact_for_review.zip HEAD
```

## 4) Checklist before sharing
- [ ] No names/emails/domains in files or notebook metadata
- [ ] README and dataset cards are neutral and anonymous
- [ ] Anonymous link works in a private browser session
- [ ] Optional: tests pass (`pytest -q`)
