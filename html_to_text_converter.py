import requests
from bs4 import BeautifulSoup
import json
import argparse

def fetch_html_content(url: str) -> str | None:
    """
    Fetches HTML content from a given URL.

    Args:
        url: The URL to fetch HTML from.

    Returns:
        The HTML content as a string, or None if an error occurs.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def extract_text_from_html(html_content: str) -> str:
    """
    Parses HTML content and extracts all visible text.
    If the marker "1.\nPDF" is found, text extraction starts from this marker.
    Otherwise, an empty string is returned.

    Args:
        html_content: The HTML content as a string.

    Returns:
        A single string containing all extracted text starting from the marker, 
        or an empty string if the marker is not found.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script, style, and other non-visible elements
    for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
        element.decompose()

    # Get text
    text = soup.get_text(separator='\n', strip=True)
    
    # Further clean up: break into lines and remove leading/trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Drop blank lines
    cleaned_text = '\n'.join(line for line in lines if line)

    print("Info: Extracted text from HTML content.", cleaned_text)

    # start_marker = "1.\nPDF"  # Define the marker to look for
    start_marker = "1."  # Define the marker to look for
    try:
        # Find the index of the start marker
        start_index = cleaned_text.index(start_marker)
        # Return the substring from the marker onwards
        print(f"Info: Start marker '{start_marker}' found. Extracting text from this point.")
        return cleaned_text[start_index:]
    except ValueError:
        # Marker not found
        print(f"Info: Start marker '{start_marker}' not found in the extracted text. Returning empty string.")
        return ""

def search_duckduckgo_lite(query_string: str, region: str = "wt-wt") -> str | None:
    """
    Performs a search on DuckDuckGo Lite using a POST request, similar to the curl command.

    Args:
        query_string: The search query.
        region: The region for the search (e.g., "wt-wt", "us-en").

    Returns:
        The HTML content of the search results page, or None if an error occurs.
    """
    url = "https://lite.duckduckgo.com/lite"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    cookies = {
        'kl': region
    }
    data = {
        'q': query_string
    }
    try:
        print(f"Searching DuckDuckGo Lite for: \"{query_string}\" with region: \"{region}\"")
        response = requests.post(url, headers=headers, cookies=cookies, data=data, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error searching DuckDuckGo Lite for \"{query_string}\": {e}")
        return None

def convert_url_to_text_json(url: str) -> str | None:
    """
    Fetches HTML from a URL, extracts text, and returns it as a JSON string.

    Args:
        url: The URL to process.

    Returns:
        A JSON string containing the URL and extracted text, or None if an error occurs.
    """
    print(f"Fetching HTML from: {url}")
    html = fetch_html_content(url)
    if html:
        print("HTML fetched successfully. Extracting text...")
        extracted_text = extract_text_from_html(html)
        print("Text extracted successfully.")
        output_data = { # Changed variable name from output_json to output_data
            "source_type": "url",
            "source_value": url,
            "extracted_text": extracted_text
        }
        return json.dumps(output_data, indent=4, ensure_ascii=False)
    return None

def main():
    parser = argparse.ArgumentParser(description="Fetch HTML from a URL or search DuckDuckGo Lite, extract text, and output as JSON.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="The URL to fetch and process.")
    group.add_argument("--search-lite", help="The query string to search on DuckDuckGo Lite.")

    parser.add_argument("--region", default="wt-wt", help="Region for DuckDuckGo Lite search (e.g., 'us-en'). Default is 'wt-wt'.")
    parser.add_argument("-o", "--output", help="Optional. File path to save the JSON output.")

    args = parser.parse_args()

    result_json = None
    source_identifier_for_error_msg = "the provided input"

    if args.url:
        source_identifier_for_error_msg = args.url
        print(f"Processing URL: {args.url}...")
        html_content = fetch_html_content(args.url)
        if html_content:
            print("HTML fetched successfully. Extracting text...")
            extracted_text = extract_text_from_html(html_content)
            print("Text extracted successfully.")
            output_data = {
                "source_type": "url",
                "source_value": args.url,
                "extracted_text": extracted_text
            }
            result_json = json.dumps(output_data, indent=4, ensure_ascii=False)

    elif args.search_lite:
        source_identifier_for_error_msg = f"DuckDuckGo Lite search: '{args.search_lite}'"
        print(f"Processing DuckDuckGo Lite search: '{args.search_lite}' with region: '{args.region}'...")
        html_content = search_duckduckgo_lite(args.search_lite, args.region)
        if html_content:
            print("Search successful. Extracting text from results...")
            extracted_text = extract_text_from_html(html_content)
            print("Text extracted successfully.")
            output_data = {
                "source_type": "duckduckgo_lite_search",
                "query": args.search_lite,
                "region": args.region,
                "extracted_text": extracted_text
            }
            result_json = json.dumps(output_data, indent=4, ensure_ascii=False)
    
    if result_json:
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result_json)
            print(f"Output saved to {args.output}")
        else:
            print(result_json)
    else:
        print(f"Failed to process: {source_identifier_for_error_msg}")

if __name__ == "__main__":
    main()
