from langgraph_sdk import get_client
import asyncio

from agent import graph

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?",
            }],
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())

from IPython.display import Image, display
# Show the butler's thought process
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

