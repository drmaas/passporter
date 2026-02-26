# Passporter

Passport photo validator and processor built with Streamlit.

The app lets you:
- Upload a selfie
- Remove the background
- Validate basic face/photo constraints
- Export a 2x2 style passport photo

## Run locally

### Prerequisites
- Python 3.11+
- Git
- One of:
  - uv (recommended)
  - pip + virtualenv

### Option A: uv (recommended)
1. Clone and open the project:

	```bash
	git clone https://github.com/<your-username>/passporter.git
	cd passporter
	```

2. Create the virtual environment and install dependencies:

	```bash
	uv sync
	```

3. Run the app:

	```bash
	uv run streamlit run app.py
	```

4. Open the local URL shown in terminal (usually http://localhost:8501).

### Option B: pip + venv
1. Clone and open the project:

	```bash
	git clone https://github.com/<your-username>/passporter.git
	cd passporter
	```

2. Create and activate a virtual environment:

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

3. Install dependencies:

	```bash
	pip install -e .
	```

4. Run the app:

	```bash
	streamlit run app.py
	```

## Deploy to share.streamlit.io

This uses Streamlit Community Cloud.

1. Push this project to a public GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click New app.
4. Choose:
	- Repository: your GitHub repo
	- Branch: main (or your deploy branch)
	- Main file path: app.py
5. Click Deploy.

After build completes, Streamlit gives you a public app URL.

## Notes

- This project pins MediaPipe to a version that includes Face Mesh solutions support.
- First run may take longer due to model/package initialization.
- If deployment fails due to dependencies, trigger a redeploy from the app settings after verifying pyproject.toml is up to date.
