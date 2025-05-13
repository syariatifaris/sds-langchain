# langchain_llama_pdf_search.py
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
import os
import csv
import json
import time

from html_to_text_converter import search_duckduckgo_lite, extract_text_from_html

from langchain_google_community import GoogleSearchAPIWrapper
search = GoogleSearchAPIWrapper()

# === Tool: Google Search ===
@tool
def google_search(query: str) -> str:
    """Searches Google and returns top 3 results."""
    results = search.results(query, 3)
    output = []

    print(f"✅ Google search results for query {query}: {results}")
    for r in results:
        output.append(f"{r['title']} - {r['link']}\n{r['snippet']}")
    return "\n\n".join(output)
            

# === Tool: Get PDF content ===
@tool
def get_pdf_content() -> str:
    """Returns the entire PDF content."""
    return pdf_text

# === Tool: Search DuckDuckGo ===
@tool
def duckduckgo_search(query: str) -> str:
    """Searches DuckDuckGo and returns top 3 results."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        output = []
        for r in results:
            output.append(f"{r['title']} - {r['href']}\n{r['body']}")
        return "\n\n".join(output)

# === Tool: Search DuckDuckGo Lite ===
@tool
def duckduckgo_lite_search(query: str) -> str:
    """Searches DuckDuckGo Lite and returns extracted text from the search results."""
    html_content = search_duckduckgo_lite(query)
    if html_content:
        return extract_text_from_html(html_content)
    return "No content found from DuckDuckGo Lite search."

# === LLM: OpenAI ===
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Using GPT-4 for better Danish language understanding
    temperature=0
)

# === Memory for conversation context ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === Initialize Agent ===
tools = [
    Tool(
        name="get_pdf_content",
        description="Returns the entire PDF content.",
        func=get_pdf_content
    ),
    Tool(
        name="duckduckgo_search",
        description="Searches DuckDuckGo and returns top 3 results.",
        func=duckduckgo_search
    ),
    Tool(
        name="google_search",
        description="Searches Google and returns top 3 results.",
        func=google_search
    ),
    Tool(
        name="duckduckgo_lite_search",
        description="Searches DuckDuckGo Lite and returns extracted text from the search results.",
        func=duckduckgo_lite_search
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    memory=memory,
    handle_parsing_errors=True
)

# === Iterate through PDFs and write to CSV ===
output_csv = "output.csv"
processed_file_csv = "processed_file.csv"
pdfs_folder = "./pdfs"
json_folder = "./pdfs-json"

# Prepare CSV file with updated headers only if it doesn't exist
if not os.path.exists(output_csv):
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["document","product_name", "manufacturer_supplier", "current_sds_version", "current_sds_date", "latest_sds_url", "latest_sds_version", "latest_sds_date"])

# Prepare processed_file_csv with updated headers only if it doesn't exist
if not os.path.exists(processed_file_csv):
    with open(processed_file_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["document"])

# Process each PDF file
for pdf_file in sorted(os.listdir(pdfs_folder)):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdfs_folder, pdf_file)

        response1 = {
            "output": json.dumps({
                "product_code": "",
                "product_name": "",
                "manufacturer_supplier": "",
                "product_item_number": "",
                "ufi_code": "",
                "current_sds_version": "",
                "current_sds_date": "",
                "language_country": "",
                "intended_use": ""
            })
        }
        product_info = response1["output"].strip()

        response2 = {
            "output": json.dumps({
                "latest_sds_url": "",
                "latest_sds_version": "",
                "latest_sds_date": ""
            })
        }

        isPDFValid = False
        
        try:
            # Load PDF content
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            pdf_text = " ".join([page.page_content for page in pages])
            isPDFValid = True
        except Exception as e:
            print(f"🚨 Error loading PDF {pdf_file}: {e}")
            isPDFValid = False

        if isPDFValid:
            # Update the get_pdf_content tool to use the current PDF content
            @tool
            def get_pdf_content() -> str:
                """Returns the entire PDF content."""
                return pdf_text

            # === Step 1: Extract Product Information ===
            print(f"\n🧠 Processing file: {pdf_file}")
            print("\n🧠 Step 1: Extracting product information...")

            # Skipped if already has the json
            json_filename = os.path.splitext(pdf_file)[0] + "_1.json"
            json_path = os.path.join(json_folder, json_filename)
            sanitized_product_info = ""

            if not os.path.exists(json_path):

                response1 = agent.invoke({
                    "input": """First, read the document. Then find the potential product information corresponding to the document. 
                    The document is an SDS document, written in Danish. Write a response in JSON format such as:
                    {
                        \"product_code\": \"(e.g., ABC)\",
                        \"product_name\": \"(e.g., Orius 200 EW)\",
                        \"manufacturer_supplier\": \"(e.g., ADAMA)\",
                        \"product_item_number\": \"16114071\",
                        \"ufi_code\": \"(e.g., XXXX-XXXX-XXXX-XXXX)\",
                        \"current_sds_version\": \"(e.g., 1.0)\",
                        \"current_sds_date\": \"(e.g., 27 October 2015)\",
                        \"language_country\": \"(e.g., Danish / Denmark)\",
                        \"intended_use\": \"(e.g., Herbicide for cereal crops)\"
                    }
                    If one or some data is not available, just fill with empty string. Return the response as a string, not a JSON object."""
                })
                print("\n📝 Response 1:\n", response1["output"])

                # Store the product information
                product_info = response1["output"].strip()
                memory.chat_memory.add_user_message(f"Product information: {product_info}")

                # Sanitize product_info to escape double quotes
                sanitized_product_info = product_info.replace('"', '\"')

                # Debug: Log sanitized_product_info before invoking the agent
                print(f"\n🔍 Debug: Sanitized product info: {sanitized_product_info}")

                # Save to json using filename
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(product_info, json_file, ensure_ascii=False, indent=4)
                print(f"🔍 Debug: Product information saved to {json_path}")
            else:
                with open(json_path, "r", encoding="utf-8") as json_file:
                    product_info = json.load(json_file)
                    # Sanitize product_info to escape double quotes
                    sanitized_product_info = product_info.replace('"', '\"')
                    print(f"🔍 Debug: Product information loaded from {json_path}")
            
            # === Step 2: Follow-up (use memory) ===
            print("\n🧠 Step 2: Searching for information...")
            json_filename = os.path.splitext(pdf_file)[0] + "_2.json"
            print(f"json_filename: Searching for {json_filename}")
            json_path = os.path.join(json_folder, json_filename)

            hasValidJson = False
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as json_file:
                    response2 = json.load(json_file)
                    try:
                        # Check if response2["output"] is valid JSON
                        if response2["output"].strip().startswith("{") and response2["output"].strip().endswith("}"):
                            # Parse response2 JSON
                            response2_dict = json.loads(response2["output"])
                            latest_sds_url = response2_dict.get("latest_sds_url", "")
                            latest_sds_version = response2_dict.get("latest_sds_version", "")
                            latest_sds_date = response2_dict.get("latest_sds_date", "")
                            hasValidJson = True
                        else:
                            print(f"🚨 Invalid JSON in response2['output'] for file {pdf_file}: {response2['output']}")
                            hasValidJson = False
                    except Exception as e:
                        print(f"🚨 Error parsing response2['output'] for file {pdf_file}: {e}")
                        hasValidJson = False

            if not hasValidJson:
                try:
                    response2 = agent.invoke({
                        "input": f"""use duckduckgo_lite_search to find information about {sanitized_product_info}. 
                        Find the newest version of SDS URL and return it in JSON format:
                        {{
                            \"latest_sds_url\": \"(e.g., https://domain.com/something.pdf)\",
                            \"latest_sds_version\": \"(e.g., 2.0)\",
                            \"latest_sds_date\": \"(e.g., 27 October 2015)\",
                        }}
                        If one or some data is not available, just fill with empty string. 
                        Return the response as a string, not a JSON object."""
                    })
                
                    # write to json using filename
                    with open(json_path, "w", encoding="utf-8") as json_file:
                        json.dump(response2["output"], json_file, ensure_ascii=False, indent=4)
                    print(f"🔍 Debug: Response 2 saved to {json_path}")
                except Exception as e:
                    print(f"🚨🚨🚨 Error during agent invocation for file {pdf_file}: {e}")
                    # Set default values in case of error
                    error_message = str(e).replace("\n", "\\n")
                    response2 = {
                        "output": json.dumps({
                            "latest_sds_url": f"🚨🚨🚨 ERROR: {error_message}",
                            "latest_sds_version": "",
                            "latest_sds_date": ""
                        })
                    }
                    # continue
            else:
                with open(json_path, "r", encoding="utf-8") as json_file:
                    response2 = {
                        "output": json.load(json_file)
                    }
                    print(f"🔍 Debug: Response 2 loaded and parsed from {json_path}: {response2}")

            print("\n📝 Response 2:\n", response2["output"])

            # === Step 3: Write to Processed CSV ===
            with open(processed_file_csv, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([pdf_file])
                print(f"🔍 Debug: Processed file {pdf_file} written to {processed_file_csv}")

        # === Optional: View memory log ===
        # print("\n📜 Agent Memory:\n", memory.buffer)

        # Parse product_info JSON
        try:
            product_info_dict = json.loads(product_info)
            manufacturer_supplier = product_info_dict.get("manufacturer_supplier", "")
            product_name = product_info_dict.get("product_name", "")
            current_sds_version = product_info_dict.get("current_sds_version", "N/A")
            current_sds_date = product_info_dict.get("current_sds_date", "N/A")  # Assuming this field contains the date

            # Parse response2 JSON
            response2_dict = json.loads(response2["output"])
            latest_sds_url = response2_dict.get("latest_sds_url", "")
            latest_sds_version = response2_dict.get("latest_sds_version", "")
            latest_sds_date = response2_dict.get("latest_sds_date", "")

            # Debug: Log data before attempting to write to CSV
            print(f"\n🔍 Debug: Preparing to write to CSV: {product_name}, {manufacturer_supplier}, {current_sds_version}, {current_sds_date}, {latest_sds_url}, {latest_sds_version}, {latest_sds_date}")

            # Write to CSV
            with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    pdf_file,
                    product_name,
                    manufacturer_supplier,
                    current_sds_version,
                    current_sds_date,
                    latest_sds_url,
                    latest_sds_version,
                    latest_sds_date
                ])
        except Exception as e:
            print(f"Error writing to CSV for file {pdf_file}: {e}")

        # Clear agent memory after processing the current PDF file
        memory.chat_memory.clear()

print(f"\n✅ Processing complete. Results saved to {output_csv}")
