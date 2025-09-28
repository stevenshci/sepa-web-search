import os
import asyncio
import aiohttp
import trafilatura
import yaml
from langchain.docstore.document import Document


async def search_web(query, embedding_model, cross_encoder, trusted=True, include_video=False, youtube_only=False):
    """Search the web using Google Custom Search API with improved efficiency."""

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    trusted_domains = []
    for category in config['trusted_domains'].values():
        trusted_domains.extend(category)

    async with aiohttp.ClientSession() as session:
        if youtube_only:
            site_query = " site:youtube.com"
            num_results = 3

            cleaned_query = query.replace('"', '').replace("'", "")
            for term in ["video", "show me", "give me", "can you", "please", "highlight"]:
                cleaned_query = cleaned_query.replace(term, "")

            if "how to" not in cleaned_query.lower() and not any(term in cleaned_query.lower() for term in ["highlight", "game", "match"]):
                instructional_query = f"how to {cleaned_query.strip()}"
                full_query = f"{instructional_query}{site_query}"
            else:
                full_query = f"{cleaned_query.strip()}{site_query}"

        elif trusted:
            top_domains = [
                "health.harvard.edu",
                "bjsm.bmj.com",
                "gssiweb.org",
                "med.stanford.edu",
                "mayoclinic.org",
                "hopkinsmedicine.org",
                "ncaa.org/sports-science",
                "strengthandconditioning.org",
                "stopsportsinjuries.org"
            ]

            if include_video:
                top_domains.append("youtube.com")
                num_results = 5
            else:
                num_results = 7

            site_query = f" site:{' OR site:'.join(top_domains)}"
            full_query = f"{query}{site_query}"
        else:
            site_query = ""
            num_results = 4
            full_query = query

        full_query = full_query.replace('"', '').replace("'", "")

        # Set up the API parameters
        params = {
            'q': full_query,
            'key': os.environ.get("GOOGLE_API_KEY"),
            'cx': os.environ.get("GOOGLE_CSE_ID"),
            'num': num_results
        }

        # Execute the search
        async with session.get("https://www.googleapis.com/customsearch/v1", params=params) as response:
            if response.status == 200:
                results = await response.json()
                items = results.get('items', [])

                # For YouTube-only search, filter to ensure we only get valid YouTube links
                if youtube_only:
                    valid_items = []
                    for item in items:
                        # Check for proper YouTube watch URLs or shortened youtu.be links
                        if ('youtube.com/watch' in item['link'] and 'v=' in item['link']) or ('youtu.be/' in item['link']):
                            # Further filter out any suspicious fake URLs
                            if not any(fake in item['link'] for fake in ['videoid', 'watch?v=123', 'yourVideoID', 'VIDEOID']):
                                valid_items.append(item)

                    items = valid_items

                return items
            else:
                return []


async def fetch_content(url):
    # Don't process YouTube URLs - they're handled separately
    if 'youtube.com' in url or 'youtu.be' in url:
        return f"Video content available at {url}"

    try:
        timeout = aiohttp.ClientTimeout(total=8)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout,
                                headers={'User-Agent': 'Mozilla/5.0'}) as response:
                if response.status != 200:
                    return None
                html = await response.text()

            extracted = await asyncio.to_thread(
                trafilatura.extract,
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True
            )

            if extracted and len(extracted) > 100:
                return extracted[:2500]  # Limit content length
            else:
                return None

    except Exception:
        return None


async def process_documents(results, query, embedding_model, cross_encoder, text_splitter, search_params):

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    trusted_domains = []
    for category in config['trusted_domains'].values():
        trusted_domains.extend(category)

    docs = []
    fetched_docs = []

    # Maximum number of documents to process (for efficiency)
    MAX_DOCS = 4

    for result in results:
        is_youtube = 'youtube.com' in result['link'] or 'youtu.be' in result['link']

        # For YouTube links, we don't need to fetch content
        if is_youtube:
            # Extract video ID for better display
            video_id = None
            if 'youtube.com/watch' in result['link'] and 'v=' in result['link']:
                video_id = result['link'].split('v=')[1].split('&')[0]
            elif 'youtu.be/' in result['link']:
                video_id = result['link'].split('youtu.be/')[1].split('?')[0]

            raw_title = result.get('title', '')
            video_title = raw_title.replace(' - YouTube', '').strip()
            if not video_title:
                video_title = f"Video about {query}"

            # Create enhanced content description with title prominently featured
            content_desc = f"""
RECOMMENDED YOUTUBE VIDEO: "{video_title}"
URL: {result['link']}
This video has been evaluated and found to be relevant to the query about {query}.
When referring to this video in your response, mention its title and indicate that it's a helpful resource.
"""

            fetched_docs.append({
                "link": result['link'],
                "title": video_title,
                "content": content_desc,
                "is_video": True,
                "video_id": video_id,
                "display_title": video_title
            })
            continue

        if not any(trusted in result['link'] for trusted in trusted_domains):
            continue

        content = await fetch_content(result['link'])
        if content:
            fetched_docs.append({
                "link": result['link'],
                "title": result.get('title', 'Untitled'),
                "content": content,
                "is_video": False
            })

            if len(fetched_docs) >= MAX_DOCS:
                break

    if not fetched_docs:
        return []

    # Re-rank using cross-encoder (only for non-video content)
    text_docs = [doc for doc in fetched_docs if not doc.get("is_video", False)]
    if text_docs:
        # Offload CPU-bound cross-encoder prediction to thread pool
        async def predict_relevance(text_docs):
            re_rank_inputs = [(query, doc["content"]) for doc in text_docs]
            return await asyncio.to_thread(cross_encoder.predict, re_rank_inputs)

        scores = await predict_relevance(text_docs)

        score_idx = 0
        for doc in fetched_docs:
            if not doc.get("is_video", False):
                doc["score"] = float(scores[score_idx])
                score_idx += 1
            else:
                doc["score"] = 0.75

        fetched_docs = sorted(fetched_docs, key=lambda x: x["score"], reverse=True)

    for doc_item in fetched_docs:
        if doc_item.get("is_video", False):
            docs.append(
                Document(
                    page_content=doc_item["content"],
                    metadata={
                        'source': doc_item["link"],
                        'title': doc_item["title"],
                        'score': doc_item.get("score", 0.5),
                        'type': 'video',
                        'display_title': doc_item.get("display_title", doc_item["title"]),
                        'video_id': doc_item.get("video_id", "")
                    }
                )
            )
            continue

        chunks = await asyncio.to_thread(text_splitter.split_text, doc_item["content"])

        docs.extend([
            Document(
                page_content=chunk[:search_params['max_content_length']],
                metadata={
                    'source': doc_item["link"],
                    'title': doc_item["title"],
                    'score': doc_item.get("score", 0.5),
                    'type': 'text'
                }
            )
            for chunk in chunks[:2] if len(chunk.strip()) > 20
        ])

    # Limit total number of documents returned
    return docs[:MAX_DOCS * 2]  # Allow for multiple chunks per source