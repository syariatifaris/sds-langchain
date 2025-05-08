# langchain_llama_pdf_search.py
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from duckduckgo_search import DDGS
from langchain_ollama import ChatOllama

# === Load PDF ===
loader = PyPDFLoader("sample.pdf")  # Replace with your file path
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

# === LLM: LLaMA 3 via Ollama ===
llm = ChatOllama(model="llama3", temperature=0)

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
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# === Step 1: Extract Product ID and search ===
print("\n🧠 Step 1: Extracting name and searching...")
response1 = agent.invoke({"input": "First, use the get_pdf_content tool to read the PDF. Then, analyze the content to find the name mentioned in the document. return the name to response."})
print("\n📝 Response 1:\n", response1["output"])

# Extract name from response1
name = response1["output"].strip()
memory.chat_memory.add_user_message(f"The person name is {name}")

# === Step 2: Follow-up (use memory) ===
print("\n🧠 Step 2: Searching for information...")
response2 = agent.invoke({"input": f"Use duckduckgo_search to find information about {{name}}. And tell me the education (university name) of the {{name}}. Then stop the operation."})
print("\n📝 Response 2:\n", response2["output"])

# === Optional: View memory log ===
print("\n📜 Agent Memory:\n", memory.buffer)
