# --- File: Final_Final.py ---
import os
import requests
import pandas as pd
import json
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
from typing import Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from langsmith import traceable
from langgraph.graph import StateGraph
import streamlit as st

# Load .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_key = os.getenv("SUPABASE_KEY")
supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_key)

# Load and preprocess dataset
music_data = pd.read_csv("Tracks_and_Audio_Features_Dataset.csv")
features = ["artists", "track_name", "popularity", "duration_ms", "explicit", "danceability", "energy", "key",
            "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature"]
music_data = music_data[features]
clustering_features = ["popularity", "duration_ms", "explicit", "danceability", "energy", "key",
                       "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(music_data[clustering_features])
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
music_data['cluster'] = kmeans.fit_predict(scaled_features)
mood_mapping = {0: 'Energetic', 1: 'Chill', 2: 'Romantic', 3: 'Happy', 4: 'Sad', 5: 'Party'}
music_data['Mood_Level'] = music_data['cluster'].map(mood_mapping)

# Agents
class StartAgent:
    def run(self, state):
        st.title("🎵 Welcome to Our Mood-Based Music Recommendation Agent!🎶")

        st.markdown(
            """
            _Discover music that matches your current mood. Whether you're feeling happy, chill, or ready to party,
            our smart AI-powered system recommends the perfect tracks for you — and remembers your preferences too!_
            """,
            unsafe_allow_html=True
        )

        if "start_clicked" not in st.session_state:
            st.session_state.start_clicked = False

        if not st.session_state.start_clicked:
            if st.button("Let's get started"):
                st.session_state.start_clicked = True
            else:
                st.stop()

        return state

class HumanInputAgent:
    def run(self, state):
        user_id = st.session_state.get("user_id")
        if not user_id:
            user_input = st.text_input("Enter your user ID (or type 'new'):")
            if user_input.lower() == "new":
                user_id = str(uuid.uuid4())
                st.session_state.user_id = user_id
                st.success(f"Your new user ID is: {user_id}")
            elif user_input:
                user_id = user_input
                st.session_state.user_id = user_id
            else:
                st.stop()

        # Clear User's Memory Button
        if st.button("Clear My Memory"):
            memory_agent.clear_user_memory(user_id)
            st.success("Your memory has been cleared.")
            st.session_state.clear()  # Reset session
            st.rerun()  # Restart app

        mood = st.selectbox("Select your mood:", ['Happy', 'Sad', 'Energetic', 'Chill', 'Romantic', 'Party'])
        if st.button("Continue"):
            state["user_id"] = user_id
            state["mood"] = mood
            state["memory"] = memory_agent.memory
            return state
        st.stop()

class MoodBasedRecommendationAgent:
    def run(self, state):
        mood = state["mood"]
        user_id = state["user_id"]
        memory = state.get("memory", {})
        prev_songs = memory.get(user_id, {}).get(mood.lower(), [])
        seen = {(s["track_name"].lower(), s["artists"].lower()) for s in prev_songs}
        mood_data = music_data[music_data["Mood_Level"] == mood]
        filtered = mood_data[~mood_data.apply(lambda row: (str(row["track_name"]).lower(), str(row["artists"]).lower()) in seen, axis=1)]
        sample_size = min(15, len(filtered))
        if sample_size == 0:
            filtered = mood_data
            sample_size = min(15, len(filtered))
        state["filtered_songs"] = filtered.sample(n=sample_size).to_dict(orient="records")
        return state

class AIRecommendationAgent:
    @traceable
    def run(self, state):
        mood = state["mood"]
        songs = state["filtered_songs"]
        prompt = f"Given this list of songs:\n{json.dumps(songs, indent=2)}\nSelect exactly 10 songs for a '{mood}' mood."
        try:
            headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"}
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
        except:
            reply = "\n".join([f"{i+1}. {s['track_name']} - {s['artists']}" for i, s in enumerate(songs[:10])])
        state["ai_recommendation"] = reply
        return state

class MemoryAgent:
    def __init__(self, url, key):
        self.supabase = create_client(url, key)
        self.memory_file = "user_memory.json"
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return json.load(f)
        return {}

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f)

    def push_to_supabase(self, uid, mood, recs):
        rows = [{"user_id": uid, "mood": mood, "track_name": s["track_name"], "artists": s["artists"]} for s in recs]
        self.supabase.table("user_memo").insert(rows).execute()

    def clear_user_memory(self, user_id):
        """Clear all memory for the given user ID, both locally and from Supabase."""
        if user_id in self.memory:
            del self.memory[user_id]
            self.save_memory()
        try:
            self.supabase.table("user_memo").delete().eq("user_id", user_id).execute()
        except Exception as e:
            print(f"Failed to delete from Supabase: {e}")

    def run(self, state):
        uid, mood, recs = state["user_id"], state["mood"].lower(), state["ai_recommendation"]
        songs = [l.split(". ", 1)[-1] for l in recs.split("\n") if l.strip() and l[0].isdigit()]
        parsed = [{"track_name": s.split(" - ")[0], "artists": s.split(" - ")[1] if " - " in s else "Unknown"} for s in songs]
        self.memory.setdefault(uid, {}).setdefault(mood, []).extend(parsed)
        self.save_memory()
        self.push_to_supabase(uid, mood, parsed)
        state["memory"] = self.memory
        return state

class OutputAgent:
    def run(self, state):
        user_id = state.get("user_id")
        user_memory = state.get("memory", {}).get(user_id, {})
        st.subheader("🎶 AI-Powered Recommendations")
        st.text(state.get("ai_recommendation", "None"))
        st.subheader(f"📦 Memory Snapshot for User: {user_id}")
        st.json(user_memory)
        return state

# Setup memory agent
memory_agent = MemoryAgent(SUPABASE_URL, SUPABASE_key)

# Build LangGraph
workflow = StateGraph(Dict[str, Any])
workflow.add_node("start", StartAgent().run)
workflow.add_node("input", HumanInputAgent().run)
workflow.add_node("filter", MoodBasedRecommendationAgent().run)
workflow.add_node("ai_recommendation", AIRecommendationAgent().run)
workflow.add_node("memory", memory_agent.run)
workflow.add_node("output", OutputAgent().run)
workflow.set_entry_point("start")
workflow.add_edge("start", "input")
workflow.add_edge("input", "filter")
workflow.add_edge("filter", "ai_recommendation")
workflow.add_edge("ai_recommendation", "memory")
workflow.add_edge("memory", "output")
executor = workflow.compile()
