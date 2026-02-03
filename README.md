# ArxivMiner

Daily arXiv paper collector with Chinese translation and auto-tagging.

## Features

- Automated daily arXiv paper collection
- English to Chinese translation (via DeepL API)
- Automatic tag generation using keyword extraction
- Web-based browsing interface
- Search, filter, and sort papers
- Mark papers as read/favorite/hidden
- RESTful API

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml` with your settings:

```yaml
# arXiv Categories to monitor
arxiv_categories:
  - cs.AI
  - cs.CL
  - stat.ML

# DeepL API Key for translation
deepl_api_key: your_api_key_here
```

### 3. Run

```bash
python run.sh
```

The app will be available at http://localhost:8000

## API Endpoints

- `GET /api/papers` - List papers with filtering
- `GET /api/papers/{arxiv_id}` - Get paper details
- `POST /api/papers/{arxiv_id}/state` - Update user state
- `GET /api/tags` - List all tags
- `POST /api/admin/fetch` - Trigger manual fetch (requires admin token)
- `GET /api/admin/stats` - System statistics

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `arxiv_categories` | arXiv categories to monitor | `["cs.AI", "cs.CL", "stat.ML"]` |
| `fetch_window_hours` | Hours to look back for new papers | `24` |
| `deepl_api_key` | DeepL API key for translation | `""` |
| `tags_per_paper` | Number of tags per paper | `8` |
| `admin_token` | Admin API token | `"admin_secret_token"` |

## License

MIT
