# Financial Data Chatbot

This project implements a chatbot capable of answering financial queries using structured and unstructured datasets. The chatbot utilizes semantic search with FAISS and generates responses using OpenAI's GPT API. It supports CSV, PDF, and PowerPoint files as data sources.

## Features

- **Semantic Search**: Uses FAISS for vector-based retrieval.
- **Multi-Format Support**: Processes CSV, PDF, and PPT files.
- **AI-Generated Responses**: Generates contextual responses using OpenAI GPT.
- **Graph-Based Data Representation**: Supports Neo4j for relationship modeling (optional).

## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Virtual environment manager (optional but recommended)
- An OpenAI API key

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mshabanpour/test-taxgpt.git
   cd test-taxgpt
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # For Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

4. **Set OpenAI API Key**:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=<your_openai_api_key>
   ```

5. **(Optional) Set Up Neo4j**:
   - Download and install Neo4j from [Neo4j's official site](https://neo4j.com/download/).
   - Start the Neo4j server and note the URI, username, and password for configuration.

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run testbot1.py
   ```

2. Open the provided local URL in your browser (e.g., `http://localhost:8501`).

## Usage

1. **Upload Files**:
   - Click "Upload CSV" to upload a financial CSV file.
   - Click "Upload PDF" to upload financial reports in PDF format.
   - Click "Upload PPT" to upload financial presentations.

2. **Ask Questions**:
   - Enter your query in the text input field.
   - View the response generated based on the uploaded data.

## File Processing Details

- **CSV**: Each row is concatenated into a single string, and embeddings are created for each row.
- **PDF**: Extracts and processes text from all pages.
- **PPT**: Extracts and processes text from all slides.

## Key Components

- **FAISS**: Vector database for fast and efficient semantic search.
- **OpenAI API**: Generates responses based on the user query and relevant context.
- **Streamlit**: Provides an interactive frontend for the chatbot.
- **Neo4j**: (Optional) Graph database for representing relationships in data.

## Future Enhancements

- Add support for additional file types.
- Enhance Neo4j integration for advanced relationship modeling.
- Enable user authentication for secure data access.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- OpenAI for their GPT API.
- Streamlit for simplifying the development of interactive web apps.
- SentenceTransformers for efficient embedding generation.

---

For issues or feature requests, please contact [Your Name/Team] or submit an issue in the repository.

