# langchain_llama_pdf_search.py
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
import os

# === Load PDF ===
loader = PyPDFLoader("./pdfs/sample.pdf")  # Replace with your file path
pages = loader.load()
pdf_text = " ".join([page.page_content for page in pages])

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
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# === Step 1: Extract Product Information ===
print("\n🧠 Step 1: Extracting product information...")
response1 = agent.invoke({
    "input": """First, read the document. Then find the potential product information corresponding to the document. 
    The document is an SDS document, written in Danish. Write a response in JSON format such as:
    {
        "product_code": "(e.g., ABC)",
        "product_name": "(e.g., Orius 200 EW)",
        "manufacturer_supplier": "(e.g., ADAMA)",
        "product_item_number": "16114071",
        "ufi_code": "(e.g., XXXX-XXXX-XXXX-XXXX)",
        "current_sds_version": "(e.g., 27 October 2015)",
        "language_country": "(e.g., Danish / Denmark)",
        "intended_use": "(e.g., Herbicide for cereal crops)"
    }
    If one or some data is not available, just fill with empty string. Return the response as a string, not a JSON object."""
})
print("\n📝 Response 1:\n", response1["output"])

# Store the product information
product_info = response1["output"].strip()
memory.chat_memory.add_user_message(f"Product information: {product_info}")

# === Step 2: Follow-up (use memory) ===
print("\n🧠 Step 2: Searching for information...")
response2 = agent.invoke({
    "input": f"""use duckduckgo_search to find information about {product_info}. 
    Find the newest version of SDS URL and return it in JSON format:
    {{
        "latest_sds_url": "",
        "version_number": "",
        "version_date": ""
    }}
    If one or some data is not available, just fill with empty string. 
    Return the response as a string, not a JSON object."""
})
print("\n📝 Response 2:\n", response2["output"])

# === Optional: View memory log ===
print("\n📜 Agent Memory:\n", memory.buffer)
