🔍 Query Expansion System using Wikipedia & Wikidata 

A smart search enhancement system that expands user queries using both semantic embeddings and structured knowledge from Wikidata. It retrieves richer, more relevant Wikipedia documents for improved search experience.

📁 Project Structure :

├── .vscode/
│   └── launch.json
├── __pycache__/
│   ├── server.cpython-310.pyc
│   ├── query_expander.cpython-310.pyc
│   ├── server.cpython-311.pyc
│   └── query_expander.cpython-311.pyc
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── query_expander.py
├── server.py


⚙️ How It Works :

-> Frontend (HTML/CSS/JS): User enters a query in the search bar.
-> Backend (Flask in server.py): Receives query and calls query_expander.py.
-> Query Expansion:
  - Expands queries using:
    - BM25 keyword matching
    - Sentence embeddings (e.g., MiniLM)
    - Wikidata entity extraction
  - Ranks terms using a hybrid score (BM25 + Semantic Similarity).
-> Document Retrieval: Fetches and displays relevant Wikipedia URLs.
-> User Interaction: Clicking a document opens its Wikipedia page in a new tab.

📚 Key Technologies :

- Python (Flask)
- Wikipedia & Wikidata APIs
- SentenceTransformers (all-MiniLM-L6-v2)
- BM25 Scoring (rank_bm25)
- HTML, CSS, JavaScript

🔍 Features :

- Hybrid Query Expansion (Semantic + Keyword-based)
- Real-time Wikipedia document suggestions
- Clean UI for quick access
- Scalable and modular codebase

✅ Current Limitations :

- May not handle complex queries perfectly.
- Needs further domain-tuning for precision.
- Currently optimized for general English queries.

📈 Future Improvements :

- Multilingual & domain-specific support
- Real-time user feedback learning
- Personalized expansion based on search history

👨‍💻 How to Run :

Add the files and folders according to the file structure given in VScode. create an environment with your project folder and run the code using the syntax given below.
(the libraries are to be downloaded accordingly)

syntax : python -m uvicorn server:app --reload


🙏 THANK YOU 





