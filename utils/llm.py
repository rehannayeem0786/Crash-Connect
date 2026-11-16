"""
LLM-powered conversational AI for the CATS Safety Bot.
Supports Groq (primary, free & fast) and OpenRouter (fallback).
Falls back to rule-based responses when no API key is available.
"""

import os
import json
from typing import Optional, Dict, List

# === API Configuration ===
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Groq model (free tier)
GROQ_MODEL = "llama-3.3-70b-versatile"
# OpenRouter model (fallback)
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

SYSTEM_PROMPT = """You are CATS (Crisis Alert Tech System) Safety Bot — an AI assistant that monitors vehicle telemetry and safety.

Your role:
- Analyze vehicle telemetry data (speed, acceleration, impact score, battery, GPS)
- Explain risk classifications (NORMAL, WARNING, CRITICAL) in plain language
- Answer questions about vehicle safety, conditions, and alerts
- Be concise, helpful, and safety-focused
- When data is provided, reference specific numbers
- If something is CRITICAL, emphasize urgency

Classification thresholds:
- Speed: WARNING > 100 km/h, CRITICAL > 140 km/h
- Acceleration: WARNING > 4.0 g, CRITICAL > 7.0 g
- Impact Score: WARNING > 0.4, CRITICAL > 0.7
- Battery: WARNING < 30%, CRITICAL < 10%
- GPS: Loss = WARNING flag

Always ground your answers in the actual telemetry data provided. Be a vehicle safety expert."""


def get_backend() -> str:
    """Determine which LLM backend to use."""
    if GROQ_API_KEY:
        return "groq"
    elif OPENROUTER_API_KEY:
        return "openrouter"
    return "rule_based"


def _build_context(vehicle_data: Dict, stats: Optional[Dict] = None) -> str:
    """Build context string from vehicle telemetry data."""
    ctx = f"""Current Vehicle Telemetry:
- Vehicle ID: {vehicle_data.get('vehicle_id', 'Unknown')}
- Speed: {vehicle_data.get('speed', 'N/A')} km/h
- Acceleration: {vehicle_data.get('acceleration', 'N/A')} g
- Impact Score: {vehicle_data.get('impact_score', 'N/A')}
- Battery: {vehicle_data.get('battery_status', 'N/A')}%
- GPS: {'Active' if vehicle_data.get('gps_status', 1) == 1 else 'LOST'}
- Risk Classification: {vehicle_data.get('risk_label', 'N/A')}
- Timestamp: {vehicle_data.get('timestamp', 'N/A')}"""

    if stats:
        ctx += f"""

Vehicle Statistics:
- Avg Speed: {stats.get('speed', {}).get('mean', 'N/A'):.1f} km/h
- Max Speed: {stats.get('speed', {}).get('max', 'N/A'):.1f} km/h
- Avg Impact: {stats.get('impact_score', {}).get('mean', 'N/A'):.3f}
- Avg Battery: {stats.get('battery_status', {}).get('mean', 'N/A'):.1f}%"""

        if 'risk_distribution' in stats:
            ctx += f"\n- Risk Distribution: {stats['risk_distribution']}"

    return ctx


def chat_with_groq(messages: List[Dict], vehicle_data: Dict, stats: Optional[Dict] = None) -> str:
    """Send a chat request to Groq API."""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        context = _build_context(vehicle_data, stats)

        system_msg = SYSTEM_PROMPT + f"\n\n--- LIVE TELEMETRY DATA ---\n{context}"

        api_messages = [{"role": "system", "content": system_msg}]
        # Include last 10 messages for context
        for msg in messages[-10:]:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=api_messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Groq API error: {str(e)}\n\nFalling back to rule-based response."


def chat_with_openrouter(messages: List[Dict], vehicle_data: Dict, stats: Optional[Dict] = None) -> str:
    """Send a chat request to OpenRouter API."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

        context = _build_context(vehicle_data, stats)
        system_msg = SYSTEM_PROMPT + f"\n\n--- LIVE TELEMETRY DATA ---\n{context}"

        api_messages = [{"role": "system", "content": system_msg}]
        for msg in messages[-10:]:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=api_messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ OpenRouter API error: {str(e)}\n\nFalling back to rule-based response."


def chat(messages: List[Dict], vehicle_data: Dict, stats: Optional[Dict] = None) -> str:
    """
    Route chat to the best available backend.
    Priority: Groq → OpenRouter → Rule-based
    """
    backend = get_backend()

    if backend == "groq":
        return chat_with_groq(messages, vehicle_data, stats)
    elif backend == "openrouter":
        return chat_with_openrouter(messages, vehicle_data, stats)
    else:
        # Will be handled by the app's rule-based handler
        return None
