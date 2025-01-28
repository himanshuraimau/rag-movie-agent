# Netflix Movie Recommendation RAG Agent

A sophisticated movie recommendation system powered by [crewAI](https://crewai.com) that uses RAG (Retrieval Augmented Generation) to provide personalized movie and TV show recommendations from Netflix's catalog. The system combines vector search capabilities with AI agents to deliver detailed, context-aware recommendations.

## Features

- RAG-based movie and show recommendations
- Detailed content analysis and comparison
- Multi-agent system for comprehensive recommendations
- Vector search through Netflix catalog
- External web search for expanded details
- Structured report generation

## Prerequisites

- Python >=3.10 <3.13
- Ollama installed locally with llama model
- EXA Search API key (for expanded content details)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-movie-agent
```

2. Install dependencies using pip:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```properties
MODEL=ollama/llama3.2
API_BASE=http://localhost:11434
EXA_API_KEY="your-exa-api-key"
```

4. Prepare your data:
- Place your Netflix dataset in `knowledge/netflix_titles.csv`
- The system will automatically create a vector store in `chroma_db/`

## Usage

Run the recommendation system:

```bash
python -m rag_movie_agent.main
```

The system will:
1. Load and process the Netflix dataset
2. Create vector embeddings using Ollama
3. Set up the recommendation agents
4. Generate detailed recommendations based on your query
5. Output a structured report with recommendations

## System Components

- **RAG Agent**: Handles vector search and initial recommendations
- **Expand Details Agent**: Enriches recommendations with external data
- **Report Agent**: Generates the final structured report
- **Vector Store**: ChromaDB for efficient content retrieval
- **Tools**: ChromaVectorSearchTool and EXASearchTool for data access

## Configuration

- Modify `config/agents.yaml` to adjust agent behaviors
- Update `config/tasks.yaml` to customize the recommendation workflow
- Adjust vector search parameters in the tools

## Support

For issues or questions:
- [Create an issue](https://github.com/yourusername/rag-movie-agent/issues)
- [Join crewAI Discord](https://discord.com/invite/X4JWnZnxPb)
- [Documentation](https://docs.crewai.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
