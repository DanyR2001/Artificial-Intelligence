# Project Title: Mate AI one

## Overview
Final project for [81940 - artificial intelligence](https://www.unibo.it/it/studiare/dottorati-master-specializzazioni-e-altra-formazione/insegnamenti/insegnamento/2024/479022)

## Installation and Setup
Instructions on setting up the project environment:
1. Clone the repository: 
```git clone https://gitlab.com/DanieleRusso/mate-ia-one.git```

2. Start python venv: 
```
cd mate-ai-one
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies: 
- python: `pip install -r requirements.txt`
- stockfish
  - macos: `brew install stockfish`

If you want to install minimax_engine you can do:
```
pip install -e .
```

<!-- ## Data
Describe the data used in this project:
- **Raw Data**: Location and description of the raw data.
- **Processed Data**: How the raw data is processed/transformed. -->

## Usage
To run the project, from the main directory:

1. Start python venv and go into the src folder:
```
python3 -m venv venv
source venv/bin/activate
cd src
```

2. Run the server
```
python3 main.py
```

3. Open the server at the http address that is showed in the command line. It should be `http://127.0.0.1:5000` by default.

## Tests
To run tests:
```
python -m tests.main
```

To test the engine against stockfish:
```
python -m tests.test_engine
```
<!-- ## Structure
- `/data`: Contains raw and processed data.
- `/src`: Source code for the project.
  - `/scripts`: Individual scripts or modules.
  - `/notebooks`: Jupyter notebooks or similar.
- `/tests`: Test cases for your application.
- `/docs`: Additional documentation in text format (e.g., LaTeX or Word).
- `/public`: Folder where GitLab pages will write static website. 
- `index.html`: Documentation in rich format (e.g., HTML, Markdown, JavaScript), will populate `public`.
-->

## Contribution
- We don't plan to introduce client sessions for the time beging since the project is focussing on the AI and not the infra.

## Contact
If you need to contact us, please read this before doing so:
- [How to ask questions the smart way](http://www.catb.org/~esr/faqs/smart-questions.html)

Our e-mails are:
- matteo.galiazzo \[at] studio.unibo.it
