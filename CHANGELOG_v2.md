# ArxivMiner v2.0 - New Features Guide

## Overview
ArxivMiner v2.0 adds significant new functionality including semantic search, recommendations, reading lists, author tracking, and RSS feeds.

## New Features

### 1. Semantic Similarity Search
- **Embedding Support**: Optional integration with sentence-transformers or OpenAI for generating paper embeddings
- **Similar Papers**: Find papers similar to any paper based on semantic meaning
- **Cosine Similarity**: Compute similarity scores between embedding vectors
- **Configuration**: Set `embedding_provider` to "sentence-transformers" or "openai"

### 2. Personalized Recommendations
- **Hybrid Strategy**: Combines popularity, recency, and similarity scoring
- **User Preferences**: Support for preferred categories and tags
- **Trending Papers**: Find most viewed/favorited papers
- **Daily Digest**: Generate personalized paper recommendations

### 3. Reading Lists/Collections
- **Create Lists**: Organize papers into named collections
- **Public/Private**: Control visibility of lists
- **Add/Remove Papers**: Manage papers in lists
- **Position Ordering**: Order papers within lists
- **Notes**: Add personal notes to papers in lists
- **Export to BibTeX**: Export entire list as BibTeX
- **Recommendations**: Get suggestions based on list content

### 4. Author Tracking
- **Follow Authors**: Track publications by favorite researchers
- **New Paper Alerts**: Get notified when followed authors publish
- **Search Authors**: Find authors by name
- **Author Statistics**: View publication history and categories

### 5. RSS Feeds
- **Paper Feed**: Subscribe to new papers via RSS
- **Category Feeds**: Filter by arXiv category
- **Tag Feeds**: Filter by tag
- **Trending Feed**: Most viewed papers
- **Auto-discovery**: RSS autodiscovery links

### 6. Export Functionality
- **BibTeX Export**: Export single paper or multiple papers
- **JSON Export**: Export with optional embedding vectors
- **Batch Export**: Filter and export large sets
- **File Downloads**: Proper Content-Disposition headers

### 7. Enhanced Statistics
- **View Counts**: Track paper popularity
- **Favorite Counts**: Track user engagement
- **Yearly Stats**: Publication statistics by year
- **Tag Usage**: Count papers per tag

### 8. Improved Search & Filtering
- **Author Filter**: Filter papers by author name
- **Multiple Sort Options**: Sort by published, updated, views, favorites
- **Date Range**: Filter by date range
- **Pagination**: Configurable page sizes

## API Endpoints

### Papers
- `GET /api/papers` - List papers with filters
- `GET /api/papers/{arxiv_id}` - Get paper details
- `GET /api/papers/{arxiv_id}/similar` - Get similar papers
- `POST /api/papers/{arxiv_id}/state` - Update user state
- `GET /api/papers/trending` - Get trending papers
- `GET /api/papers/recommendations` - Get recommendations
- `GET /api/papers/{arxiv_id}/export/bibtex` - Export BibTeX
- `GET /api/papers/{arxiv_id}/export/json` - Export JSON
- `GET /api/papers/export/all/bibtex` - Batch export BibTeX
- `GET /api/papers/export/all/json` - Batch export JSON

### Reading Lists
- `GET /api/lists` - List reading lists
- `POST /api/lists` - Create reading list
- `GET /api/lists/{list_id}` - Get reading list
- `PUT /api/lists/{list_id}` - Update reading list
- `DELETE /api/lists/{list_id}` - Delete reading list
- `POST /api/lists/{list_id}/papers` - Add paper to list
- `DELETE /api/lists/{list_id}/papers/{paper_id}` - Remove paper
- `GET /api/lists/{list_id}/export/bibtex` - Export list BibTeX
- `GET /api/lists/{list_id}/recommendations` - Get suggestions

### Authors
- `GET /api/authors?q=...` - Search authors
- `GET /api/authors/{name}` - Get author details
- `POST /api/authors/follow` - Follow author
- `POST /api/authors/unfollow` - Unfollow author
- `GET /api/authors/followed` - List followed authors
- `GET /api/authors/new-papers` - Get new papers from followed

### RSS Feeds
- `GET /api/rss/papers` - RSS feed of papers
- `GET /api/rss/categories/{category}` - Category RSS
- `GET /api/rss/tags/{tag}` - Tag RSS
- `GET /api/rss/trending` - Trending papers RSS

## Configuration

### New Config Options
```yaml
# Embedding settings
embedding_provider: "none"  # none, sentence-transformers, openai
embedding_model: "all-MiniLM-L6-v2"
embedding_api_key: ""

# Pagination
default_page_size: 20
max_page_size: 100

# Rate limiting
rate_limit_per_minute: 60
rate_limit_per_hour: 1000

# Logging
log_level: "INFO"
```

### Environment Variables
- `OPENAI_API_KEY` - For OpenAI embeddings

## Database Schema Changes

### New Tables
- `reading_lists` - Reading list metadata
- `reading_list_items` - Papers in lists
- `author_follows` - Author subscriptions
- `user_preferences` - User settings
- `search_history` - Search queries

### New Paper Fields
- `embedding_vector` - Paper embedding
- `view_count` - View statistics
- `favorite_count` - Favorite statistics

### New UserState Fields
- `read_progress` - Reading progress (0.0-1.0)
- `last_read_at` - Last read timestamp

## Installation

```bash
# For basic functionality
pip install -r requirements.txt

# For embedding support
pip install sentence-transformers  # or
pip install openai

# For numpy (recommended for similarity)
pip install numpy
```

## Usage Examples

### Get Similar Papers
```bash
curl "http://localhost:8000/api/papers/2401.01234/similar?limit=10"
```

### Create Reading List
```bash
curl -X POST "http://localhost:8000/api/lists" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Reading List", "description": "AI papers to read", "is_public": true}'
```

### Follow an Author
```bash
curl -X POST "http://localhost:8000/api/authors/follow" \
  -H "Content-Type: application/json" \
  -d '{"author_name": "Yann LeCun", "arxiv_id": "2401.01234"}'
```

### Subscribe to RSS
```bash
# Add to your RSS reader
http://localhost:8000/api/rss/papers?limit=50
http://localhost:8000/api/rss/categories/cs.AI
http://localhost:8000/api/rss/trending?days=7
```

### Export Papers
```bash
# Download BibTeX
curl "http://localhost:8000/api/papers/export/all/bibtex?category=cs.AI&limit=100" \
  -o papers.bib

# Download JSON
curl "http://localhost:8000/api/papers/export/all/json?limit=100" \
  -o papers.json
```

## Testing

```bash
# Manual tests
python3 run_tests.py

# Unit tests (requires pytest)
python3 -m pytest tests/test_core.py -v
```

## Migration from v1.0

The database schema is backward compatible. New tables and columns are added automatically. No data migration required.

## Performance Considerations

1. **Embedding Generation**: Generate embeddings asynchronously for large batches
2. **Caching**: Cache frequently accessed data
3. **Rate Limiting**: Respect arXiv API limits
4. **Pagination**: Use reasonable page sizes (default 20, max 100)

## Future Enhancements

Planned features for future versions:
- Full-text search with Elasticsearch
- User authentication
- Email notifications
- Slack/Discord integrations
- Zotero/Mendeley sync
- Paper summarization with LLMs
- Citation graph visualization
- Mobile app
