# ORIGIN_RAG_BASED_LLM_WITH_WEB_SEARCH
An Agentic RAG system capable with webscrapping to produce NLP content and learning capabilities for the AI without pre-training. 

Origin AI Agent
An intelligent, hybrid AI agent that synthesizes knowledge by combining the power of local data analysis with real-time web intelligence. Origin can answer questions about your uploaded files and perform live web research to provide comprehensive, cited summaries on any topic.

This project is powered by Google's Gemini-1.5-Flash for its cognitive core and uses a Retrieval-Augmented Generation (RAG) architecture for its long-term memory.

Key Features
Hybrid Intelligence: Seamlessly query both local data files (like CSVs, via ask_db) and the live internet within a single conversational interface.

Knowledge Synthesis: Goes beyond simple search by ingesting information from multiple web pages and generating a clean, synthesized summary with source citations.

Resilient Web Search: Prioritizes high-quality search results using SerpApi when a key is provided, with an automatic fallback to a reliable DuckDuckGo scraper to ensure functionality is never lost.

Modular Architecture: Built with distinct, understandable components for memory (ClassicalStack), perception (WebSearch), and executive function (Controller), making the system robust and extensible.

Interactive UI: A simple and clean user interface powered by Gradio, allowing for easy interaction and testing.

Explainable & Controllable: Uses a clear command-based system (e.g., origin:, search:) for predictable behavior and provides logs for transparency.

Architecture Overview
Origin is designed like a biological agent, with specialized components for different cognitive functions:

User Input
     |
     v
[ Controller ]  (Routes the command)
     |
     +--------------------------------+----------------------------+
     v                                v                            v
[ WebSearch ]  <---> [ WebIngestor ]  [ ClassicalStack (RAG) ]   [ Hippocampus ]
(Perceives Web)      (Learns from Web)  (Long-Term Memory)        (Short-Term Memory)
     |                    ^                   |                        |
     v                    |                   v                        v
[ SerpApi / DDG ]         +------------ [ FAISS Index ]                |
                                            |                        |
                                            v                        v
                                     [ Gemini 1.5 Flash ] <----------+
                                     (Cognitive/Reasoning Core)
                                            |
                                            v
                                     Generated Output

Controller: The agent's "executive function." It parses user input and decides which action to take.

WebSearch & WebIngestor: The agent's "senses." They find and process new information from the internet.

ClassicalStack: The agent's "long-term memory." A RAG system using a FAISS vector index to store and retrieve knowledge.

Hippocampus: The agent's "short-term memory," keeping a log of recent actions for context and debugging.

Gemini 1.5 Flash: The agent's "reasoning core," used for generating answers and synthesizing summaries.

Getting Started
Follow these steps to set up and run the Origin agent on your local machine.

Prerequisites
Python 3.8 or higher

pip and git installed

1. Clone the Repository
git clone [https://github.com/your-username/origin-agent.git](https://github.com/your-username/origin-agent.git)
cd origin-agent

2. Install Dependencies
The project's dependencies are listed in requirements.txt. Install them using pip:

pip install -r requirements.txt

(Note: A requirements.txt file would include: google-generativeai, gradio, sentence-transformers, faiss-cpu, requests, beautifulsoup4, serpapi-google-search)

3. Set Up API Keys
Origin requires a Google API key to function and can be enhanced with a SerpApi key.

Google API Key (Required):

Get your key from the Google AI Studio.

Set it as an environment variable:

export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

SerpApi Key (Optional, Recommended):

Get a key from SerpApi.

You don't need to set this as an environment variable. You can enter it directly into the textbox in the Gradio UI for premium search results. If you leave it blank, the agent will automatically use the fallback search method.

4. Run the Agent
Launch the Gradio application by running the main Python script:

python agent.py

This will start a local web server. Open the URL provided in your terminal (usually http://127.0.0.1:7860) to interact with the Origin agent.

How to Use
Interact with the agent using the command bar at the bottom of the UI.

Available Commands
origin: [topic]

This is the primary command. The agent searches the web for the topic, reads the top results, and generates a comprehensive, cited summary.

Example: origin: The economic history of Costa Rica since 1980

search: [topic]

Performs a web search and ingests the content into its memory but does not generate a summary. Useful for "pre-loading" the agent with information on a topic.

Example: search: latest advancements in neural interfaces

summarize: [topic]

Generates a summary based on the most recently ingested information from a search command.

Example: summarize: neural interfaces

remember: [fact]

Stores a piece of text directly into the agent's long-term memory.

Example: remember: My favorite coffee is from the Tarrazú region.

[your question]

If no command prefix is used, the agent will answer your question based on the information it has already learned (from origin, search, or remember).

Example: What coffee region is my favorite?

Contributing
Contributions are welcome! If you have ideas for new features, bug fixes, or improvements, please feel free to open an issue or submit a pull request.

