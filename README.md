# Hate Speech Detection Assistant
The Hate Speech Detection Assistant is a generative AI-powered application designed to identify and classify user-generated text into categories such as Hate, Offensive, Toxic, Neutral, or Ambiguous. It also retrieves relevant policy documents and recommends appropriate moderation actions using explainable AI techniques.

Features
- Multiclass Classification: Detects and classifies text into five categories.
- Hybrid RAG System: Combines semantic search and keyword search to retrieve relevant policy documents.
- LLM-Based Reasoning: Explains classification using local large language models for enhanced transparency.
- Moderation Recommendations: Suggests appropriate actions based on content severity and matched policies.
- Modular Agentic Design: Each core functionality is handled by an independent agent, ensuring flexibility and scalability.
- Playground: Playground is a platform that allows users to detect hate speech in a paragraph by inspecting each sentence individually.

Architecture
- Agents:
  - HateSpeechDetectionAgent: Classifies the input text.
  - HybridRetrieverAgent: Retrieves relevant policies using FAISS and sentence-transformers.
  - PolicyReasoningAgent: Provides reasoning for classification based on retrieved content.
  - ActionRecommenderAgent: Suggests moderation actions.
  - ErrorHandlerAgent: Catches and reports errors or ambiguous inputs.

- Frontend: Built using Streamlit for an interactive user experience.
- Backend: LLMs and embedding models integrated gemini API keys.

Technologies Used
- Python
- Streamlit
- Gemini Transformers
- FAISS
- LangChain
- Langgraph
