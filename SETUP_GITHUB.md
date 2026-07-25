# Publishing `Voice-Models` to GitHub

This is the step-by-step for getting the local `D:\Voice-Models` repo onto
GitHub at `https://github.com/swayamiitb/Voice-Models`.

---

## 1. Create the empty repo on GitHub

1. Go to https://github.com/new
2. **Repository name:** `Voice-Models`
3. **Description:** `VajraVoice — a six-module open-source neural TTS engine reproducing the MAI-Voice-2 architectural class for Indian languages.`
4. Set **Public** (it's a portfolio piece).
5. **DO NOT** initialize with README / .gitignore / LICENSE — the local repo
   already has all of these, and a remote-side README would create a merge
   conflict on first push.
6. Click **Create repository**.

---

## 2. Set your git identity (if not already set)

```bash
git config --global user.name  "swayamiitb"
git config --global user.email "your-email@example.com"
```

(Use the email tied to your GitHub account so commits link to your profile.)

---

## 3. Authenticate to GitHub

You have two options:

### Option A — HTTPS with a Personal Access Token (simplest)

1. Go to https://github.com/settings/tokens → **Generate new token (classic)**
2. Scope: `repo` (full control of private/public repos)
3. Copy the token (`ghp_xxxxxxxxxxxx`)
4. When you push, git will prompt for username + password — paste the token
   as the password.
5. To cache it so you don't re-enter every push:
   ```bash
   git config --global credential.helper store
   ```

### Option B — SSH (set once, never think about it again)

1. Generate a key if you don't have one: `ssh-keygen -t ed25519 -C "your-email@example.com"`
2. Copy the public key: `cat ~/.ssh/id_ed25519.pub`
3. Add it at https://github.com/settings/keys
4. Test: `ssh -T git@github.com` → should say "Hi swayamiitb!"
5. Use the SSH remote URL below (not HTTPS).

---

## 4. Push from `D:\Voice-Models`

```bash
cd /d/Voice-Models

# (already done by the build script: git init + first commit)
git branch -M main

# Add the remote — pick ONE of these two:
git remote add origin https://github.com/swayamiitb/Voice-Models.git     # HTTPS
git remote add origin git@github.com:swayamiitb/Voice-Models.git         # SSH

# Push
git push -u origin main
```

Within ~30 seconds you should see:
- The README rendering on the repo home page
- The badge `[tests: passing]` once the first CI run completes
- All 30+ files in the file browser

---

## 5. Enable GitHub Actions (usually automatic)

- The `.github/workflows/test.yml` file is what triggers CI.
- On first push, GitHub may show a banner asking you to enable Actions for
  the repo — click **"I understand my workflows, go ahead and enable them"**.
- CI will run on every push and every PR after that. Watch it at:
  `https://github.com/swayamiitb/Voice-Models/actions`

---

## 6. Recommended follow-ups (optional, after first push)

- **Add topics** for discoverability: `text-to-speech`, `tts`, `neural-tts`,
  `indian-languages`, `voice-cloning`, `flow-matching`, `open-source`,
  `MAI-Voice-2`. Repo → ⚙️ → About → ⋯ → Topics.
- **Add a social preview image** (Repo → ⚙️ → Social preview) — pick one of
  the architecture diagrams from the docs workspace.
- **Pin the repo** to your GitHub profile.
- **Write a short release** (Releases → Draft a new release → v0.1.0) once
  CI is green.
- **Add a `CITATION.cff`** so GitHub can render a "Cite this repository"
  button. (Already included a BibTeX block in README.md.)

---

## Troubleshooting

**`git push` rejects with "non-fast-forward"** — the remote has commits the
local doesn't (usually because you initialized with a README on GitHub).
Fix: `git push --force-with-lease origin main`. (Safe on first push because
the remote is empty.)

**CI fails on `pip install -e ".[api,audio,dev]"`** — make sure
`pyproject.toml` is at the repo root and you're on Python 3.10–3.12 (3.13+
may have wheel gaps for some deps).

**Badge shows "no status"** — CI hasn't run yet. Push any commit (even an
empty one: `git commit --allow-empty -m "trigger ci" && git push`) to
kick it off.
