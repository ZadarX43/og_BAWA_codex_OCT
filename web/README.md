# Odds Genius interactive landing page

This prototype demonstrates the proposed Odds Genius landing page and fixture explorer experience. It is a static web build so it can be hosted on GitHub Pages, Netlify, Vercel, or any static file server.

## Getting started locally

1. In Finder, locate the `og_BAWA_codex_OCT` folder (for example in **Downloads** if you unzipped it there). Drag that folder somewhere convenient such as your Desktop so you know its exact location.

2. Open the **Terminal** app and navigate into the `web/` subfolder. The easiest way on macOS is to type `cd ` (with a trailing space), then drag the `web` folder from Finder into the Terminal window and press **Return**. Terminal will paste the real path for you, e.g.

    ```bash
    cd /Users/yourname/Desktop/og_BAWA_codex_OCT/web
    ```

    To double-check you are in the right directory, run `pwd` (prints your current folder) and `ls` (lists files). You should see `index.html`, `styles.css`, `app.js`, and a `data/` folder. If you see your home folders such as `Desktop/`, `Documents/`, etc., you are still in the wrong place—repeat the drag-and-drop step for the `web` folder.

3. Start a lightweight static server from that directory:

    ```bash
    python3 -m http.server 8000
    ```

4. In your browser visit `http://localhost:8000/index.html` (or simply refresh `http://localhost:8000` if the directory listing shows `index.html`). You should now see the full landing page. The Three.js globe and CSV fixture data load directly from the `data/fixtures.csv` file once the page finishes loading.

4. If the canvas stays blank, open the browser console (⌥⌘I on macOS) and confirm the CDN assets loaded. The page pulls:
   - `three.min.js`
   - `OrbitControls.js`
   - `three-globe.min.js`
   - `papaparse.min.js`

   When any of these fail to download, the globe area will show a diagnostic card listing the missing files so you can retry or host them locally.

### Troubleshooting common setup snags

* **Terminal only shows a long list of folders (Desktop, Documents, etc.) when you visit `http://localhost:8000`.** This means the server is running from your home directory instead of the `web/` folder. Stop the server with `Ctrl+C`, run the drag-and-drop `cd` step again for the `web` folder specifically, then restart `python3 -m http.server 8000`.
* **Safari/Chrome still shows the directory listing even after you moved into `web/`.** Append `/index.html` to the URL or click the `index.html` link in the listing; browsers cache the directory page until you refresh.
* **Wondering if you need to click “Create PR” in GitHub Desktop or GitHub.com?** No—local previewing doesn’t require GitHub at all. Only push to GitHub when you want to share the demo link with others.

## Hosting for free with GitHub Pages

GitHub Pages is completely free for public repositories, so you can demo the globe without paying for hosting. Once you have a GitHub account:

1. Create a new repository (or reuse an existing one) and copy the contents of the `web/` folder into the repo. The simplest approach is to commit the `web/` folder itself so `index.html` lives at the repository root or inside the `web/` directory.
2. Push the changes to GitHub.
3. In the GitHub web UI, open **Settings ▸ Pages**.
4. Choose **Deploy from a branch** and set the **Branch** dropdown to the branch that contains the landing page (usually `main`).
5. Pick the **/ (root)** option if `index.html` is at the repository root, or choose `/web` if you committed the folder without moving the files.
6. Click **Save**. Within a minute or two GitHub will publish a URL in the same Settings page (e.g. `https://<username>.github.io/<repository>`). Use that link to share the interactive prototype.

No build step or credit card is required. GitHub Pages serves the static HTML, CSS, JavaScript, and CSV files exactly as they exist in your repo.

You can follow a similar flow for Netlify or Vercel&mdash;just point the deployment at the `web/` directory and select a static site build with no build command.

## Saving your work to GitHub (kid-friendly version)

If you are new to Git and GitHub, think of it like putting your drawing on the fridge so everyone can see it. Here is the super simple flow:

1. **Tell Git which files are part of your drawing.** In Terminal (still inside the project folder) type:

   ```bash
   git add web
   ```

   This is like picking up all the papers you want to keep.

2. **Stick a label on that bundle.** Run:

   ```bash
   git commit -m "Describe your change here"
   ```

   Git now remembers exactly what changed. If it says there is nothing to commit, it means you have not edited any files yet.

3. **Push the drawing to the online fridge (GitHub).** First make sure you have created a repository on GitHub and connected it as a remote (usually called `origin`). Then run:

   ```bash
   git push origin main
   ```

   Replace `main` with the branch name you are using. GitHub now has your latest changes.

4. **(Optional) Open a Pull Request.** Visit GitHub in your browser. It will usually offer a green **Compare & pull request** button. Click it to review what changed, then press **Create pull request** if you want teammates to review or merge it. You do not need to open a PR just to preview the site locally.

That’s it—add, commit, push, and optionally make a PR. Each step is a tiny button press that moves your work from “just on my computer” to “safely saved on GitHub.”

## Data inputs

* `data/fixtures.csv` &ndash; demo dataset for Champions League fixtures (27&ndash;30 November) with xG, PPG, and player insight metrics.
* `app.js` &ndash; parses the CSV with PapaParse and renders the animated globe and insight panels.
* `styles.css` &ndash; applies the blue-green gradient palette inspired by the supplied art direction.

To plug in live data, export the latest fixtures in the same CSV structure (one row per fixture) and replace the file in `data/fixtures.csv`. The map and insight cards will automatically update on reload.

## Suggested build roadmap

1. **Wire up BetChecker OCR upload** – Replace the placeholder alert with a real file upload that posts images/PDFs to your OCR service and displays the parsed betslip output.
2. **Enrich fixture data feeds** – Automate CSV generation from your modelling pipeline (fixtures, player props, confidence scores) and include live status updates so the globe reacts to real-time changes.
3. **Club and fixture drill-ins** – Add modal/detail views for each fixture and club with historical head-to-head, form, and player trends surfaced from your data lake.
4. **OG Co-Pilot integration** – Embed a chat surface or guided prompt flow that uses your GPT-5 OG assistant to answer questions about fixtures, markets, and recommended plays.
5. **Authentication & saved workspaces** – Introduce lightweight auth so bettors can save favorite fixtures, upload histories, and revisit personalised insights.

Treat these as modular milestones—you can ship them incrementally while the static globe remains live on GitHub Pages.
