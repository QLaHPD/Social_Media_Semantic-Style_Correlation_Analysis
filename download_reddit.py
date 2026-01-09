#!/usr/bin/env python3
"""
grab_redarcs.py

Usage:
    python grab_redarcs.py https://the-eye.eu/redarcs/

Behavior:
 - Scrapes the provided index page for links ending with
   "_submissions.zst" or "_comments.zst".
 - Excludes subreddits whose base names are in the EXCLUDE list
   (case-insensitive). The EXCLUDE list below is taken from the
   screenshot you provided.
 - Calls `wget -c <url>` for each file (resumable download).
 - Downloads run in parallel (configurable workers).
"""

import sys
import re
import subprocess
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from bs4 import BeautifulSoup
except Exception as e:
    print("This script requires 'requests' and 'beautifulsoup4'.")
    print("Install: pip install requests beautifulsoup4")
    raise

# --- CONFIGURE HERE ---
EXCLUDE = [
    "AskReddit","politics","worldnews","AmItheAsshole","brasil",
    "Tinder","AMA","AskRedditAfterDark","geopolitics","anime_titties",
    "SeriousConversation"
]
# normalize to lowercase for matching
EXCLUDE_SET = {name.lower() for name in EXCLUDE}

# how many parallel downloads
WORKERS = 6

# wget options (as a list)
WGET_CMD_BASE = ["wget", "-c", "--no-clobber", "--content-disposition", "--trust-server-names"]

# ------------------------

FILE_PATTERN = re.compile(r".*/([^/]+)_(submissions|comments)\.zst$", re.IGNORECASE)

def extract_files_from_index(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # consider absolute and relative; if relative, skip (or user can give base)
        # but typical index has absolute direct links to the-eye.eu
        if href.lower().endswith(".zst"):
            m = FILE_PATTERN.match(href)
            if m:
                sub = m.group(1)
                kind = m.group(2)
                urls.append((href, sub.lower(), kind))
    return urls

def should_skip(subreddit_lower):
    # skip if the subreddit base appears in the exclusion list
    return subreddit_lower in EXCLUDE_SET

def wget_download(url):
    cmd = WGET_CMD_BASE + [url]
    print("Starting:", url)
    try:
        # run wget and wait
        p = subprocess.run(cmd, check=False)
        rc = p.returncode
        if rc == 0:
            print("Done:    ", url)
            return True, url
        else:
            print(f"wget failed (rc={rc}): {url}")
            return False, url
    except FileNotFoundError:
        print("wget not found. Install wget or modify the script to use another downloader.")
        return False, url
    except Exception as e:
        print("Error downloading", url, ":", e)
        return False, url

def main(index_url):
    print("Fetching index:", index_url)
    r = requests.get(index_url, timeout=30)
    r.raise_for_status()
    files = extract_files_from_index(r.text)

    if not files:
        print("No matching files found on the page.")
        return

    # Filter out excluded subreddits
    to_download = []
    for href, sub, kind in files:
        if should_skip(sub):
            print("Skipping (in exclude list):", sub, kind, href)
            continue
        to_download.append(href)

    if not to_download:
        print("No files to download after applying exclude list.")
        return

    print(f"Will download {len(to_download)} files (workers={WORKERS}).")
    # run parallel wget jobs
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(wget_download, url): url for url in to_download}
        for fut in as_completed(futures):
            ok, url = fut.result()
            results.append((ok, url))

    succeeded = sum(1 for ok, _ in results if ok)
    failed = len(results) - succeeded
    print(f"Finished. Success: {succeeded}, Failed: {failed}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grab_redarcs.py <index_url>")
        sys.exit(2)
    index_url = sys.argv[1]
    main(index_url)
