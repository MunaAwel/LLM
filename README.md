# LLM-Based Mood Music Recommendation System

This project presents an intelligent mood-aware music recommender powered by a Large Language Model (LLM), combining traditional machine learning techniques with modern AI orchestration tools. Users receive personalized song suggestions based on mood inputs, with their history stored for improved future recommendations. Built with modular components like LangGraph, Supabase, and Groq’s Llama 3 model, the system is scalable, flexible, and user-friendly.

## Project Summary

In a digital landscape saturated with choices, users often struggle to find songs that align with their emotional state. Our solution bridges this gap through a mood-based music recommendation system that:

- Accepts user mood and returns 10 curated song recommendations
- Stores each user's recommendation history for memory-driven personalization
- Allows users to reset or manage their memory
- Uses Groq’s Llama 3 model and K-Means clustering for recommendation generation
- Runs through an intuitive UI built with Streamlit

## Key Features

- **User-Centered Design**: Choose or generate user ID, select a mood, and receive personalized suggestions.
- **Persistent Memory**: User histories are tracked via local JSON and stored securely in Supabase.
- **AI-Driven Filtering**: Combines clustering and LLMs to optimize song matching to user moods.
- **Modular Workflow**: LangGraph orchestrates tasks including mood input, memory lookup, AI recommendation, and output.
- **Reset and Reuse**: Users can clear memory or continue refining their preferences.

## Technologies Used

- **Python**
- **Streamlit** – For the web-based interface
- **LangGraph** – To coordinate agent workflows
- **Supabase** – Cloud database for user state and history
- **Groq's Llama 3 model** – LLM used for enhanced recommendation filtering
- **Kaggle Dataset** – Used for song features and metadata
- **LangSmith** – Debugging and traceability of workflow

## System Architecture

- **Presentation Layer**: Streamlit UI for user interaction
- **Application Layer**: LangGraph for modular agent orchestration
- **Service Layer**:
  - K-Means for mood-based clustering
  - Groq API for AI-enhanced recommendation
  - Supabase for cloud memory management
- **Data Layer**: 
  - Static CSV datasets for music features
  - JSON/Supabase for persistent user memory
- **Monitoring**: LangSmith tracing for developer insights and debugging

## Challenges Encountered

- Replacing Spotify API with Kaggle datasets due to access restrictions
- Aligning file paths across Colab and local Mac environments
- Managing memory persistence between local files and Supabase
- Reconciling Streamlit’s event-driven model with LangGraph’s sequential flow
- Handling API expiration and LangSmith token issues

## Use Cases

- **Personalized Music Companion**: Recommend music based on mood without requiring a full listening history
- **Mental Health Integration**: Pair mood-based playlists with wellness or relaxation apps
- **User Mood Analytics**: Track mood trends over time for improved filtering and feedback loops
- **Developer Tooling**: Enable robust monitoring with LangSmith for continuous improvements
