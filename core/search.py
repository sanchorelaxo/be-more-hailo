import logging
try:
    from ddgs import DDGS  # new package name (pip install ddgs)
except ImportError:
    from duckduckgo_search import DDGS  # fallback for older installs

logger = logging.getLogger(__name__)

def search_web(query: str) -> str:
    """
    Searches DuckDuckGo for the given query and returns a summary of the top result.
    Special cases: Weather uses wttr.in for better accuracy.
    """
    logger.info(f"Searching web for: {query}")
    query_lower = query.lower()
    
    # 0. Special Case: Weather
    if "weather" in query_lower:
        # Extract the location by dropping the question/filler words, not by
        # requiring the word "in" — "Weather Toronto" has no "in" and used to
        # silently fall back to the default city.
        location = "Brantford"  # Default
        import re as _re
        cand = query_lower
        # Word-boundary removal so "the" doesn't mangle "weather" -> "wea r".
        cand = _re.sub(r"\b(what's|what is|whats|tell me|the|weather|forecast|"
                       r"like|today|tonight|tomorrow|please|in|at|for)\b", " ", cand)
        cand = _re.sub(r"[?!.,]", " ", cand)
        cand = " ".join(cand.split())  # collapse whitespace
        if cand:
            location = cand.replace(" ", "+")
        
        try:
            import requests
            # Compact one-line summary ("Sunny +23C 76% ...").  The v2 ASCII-art
            # table is ~16 KB of box-drawing + ANSI colour that crashes
            # hailo-ollama's strict JSON prompt renderer when injected upstream.
            url = f"https://wttr.in/{location}?format=4"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                result = f"LIVE WEATHER DATA for {location}: {resp.text.strip()}"
                logger.info(f"Weather fetched from wttr.in: {location}")
                return result
        except Exception as e:
            logger.warning(f"wttr.in Weather Error: {e}")
            # Fall through to DDG if wttr.in fails

    try:
        with DDGS(timeout=10) as ddgs:
            results = []
            
            # Use Canadian region if Ontario is mentioned or if it's a general request in this fork
            # This makes BMO feel more local to the user's setup.
            region = 'ca-en' if any(k in query_lower for k in ['ontario', 'canada', 'brantford', 'toronto']) else 'wt-wt'
            
            # 1. Try News search first for current events (skip for weather)
            if any(k in query_lower for k in ["news", "latest", "today", "happening", "current"]):
                try:
                    logger.info(f"Searching News (region={region})...")
                    results = list(ddgs.news(query, region=region, max_results=5))
                    if results:
                        logger.info(f"Found {len(results)} news items.")
                except Exception as e:
                    logger.warning(f"News Search Error: {e}")
            
            # 2. Fallback to Text search
            if not results:
                logger.info(f"Trying text search (region={region})...")
                try:
                    results = list(ddgs.text(query, region=region, max_results=3))
                    if results:
                        logger.info(f"Found Text: {results[0].get('title')}")
                except Exception as e:
                    logger.warning(f"Text Search Error: {e}")

            if results:
                # Combine up to 3 results for richer context
                parts = []
                for r in results[:3]:
                    title = r.get('title', 'No Title')
                    body = r.get('body', r.get('snippet', 'No Body'))
                    parts.append(f"Title: {title}\nSnippet: {body[:400]}")
                return f"SEARCH RESULTS for '{query}':\n" + "\n---\n".join(parts)
            else:
                logger.info("Search returned 0 results.")
                return "SEARCH_EMPTY"
                
    except Exception as e:
        logger.error(f"Connection/Library Error during search: {e}")
        return "SEARCH_ERROR"

def search_images(query: str) -> str:
    """
    Searches DuckDuckGo for the given query and returns the first image URL.
    """
    logger.info(f"Searching images for: {query}")
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.images(query, max_results=1))
            if results:
                image_url = results[0].get('image')
                logger.info(f"Found Image: {image_url}")
                return image_url
            else:
                logger.info("Image search returned 0 results.")
                return ""
    except Exception as e:
        logger.error(f"Image Search Error: {e}")
        return ""
