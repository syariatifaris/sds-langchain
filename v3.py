from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
import json  # Import the json module

search = GoogleSearchAPIWrapper()

def top3_results(query):
    return search.results(query, 3)

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=top3_results,
)

# Store the result in a variable
result = tool.run("Soudal Akryl danish sds .pdf")

# Print the result
print(result)

# Save the result to a JSON file
with open("google_result.json", "w") as json_file:
    json.dump(result, json_file)