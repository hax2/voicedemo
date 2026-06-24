"""Preset sentences for accent training demo.

Organized by language pair and difficulty level.
Each sentence includes the text and a phonetic focus note.
"""

SENTENCES = {
    "en-US": {
        "label": "English (for Spanish speakers)",
        "beginner": [
            {
                "text": "The weather is really nice today, isn't it?",
                "focus": "TH sound, contractions",
            },
            {
                "text": "She sells seashells by the seashore.",
                "focus": "SH vs S distinction",
            },
            {
                "text": "I would like a glass of water, please.",
                "focus": "W sound, soft T in water",
            },
            {
                "text": "Can you help me find the nearest bus stop?",
                "focus": "Short vowels, final consonant clusters",
            },
            {
                "text": "I've been living here for about three years.",
                "focus": "V vs B, TH sound, R sound",
            },
        ],
        "intermediate": [
            {
                "text": "Although the restaurant was crowded, we managed to get a table.",
                "focus": "Schwa sounds, word stress patterns",
            },
            {
                "text": "I thought through the problem thoroughly before answering.",
                "focus": "TH variations, -OUGH patterns",
            },
            {
                "text": "The variable results suggest further research is required.",
                "focus": "Academic vocabulary stress",
            },
            {
                "text": "Would you mind if I borrowed your phone charger?",
                "focus": "Polite intonation, linking",
            },
            {
                "text": "Statistics show that approximately sixty percent of people prefer working from home.",
                "focus": "Multi-syllable word stress, numbers",
            },
        ],
    },
    "es-ES": {
        "label": "Spanish (for English speakers)",
        "beginner": [
            {
                "text": "Buenos días, ¿cómo estás hoy?",
                "focus": "Rolling R, vowel purity",
            },
            {
                "text": "Me gustaría un café con leche, por favor.",
                "focus": "Soft G, nasal sounds",
            },
            {
                "text": "¿Dónde está la estación de tren más cercana?",
                "focus": "Question intonation, D between vowels",
            },
            {
                "text": "Mi familia vive en una casa grande cerca del parque.",
                "focus": "V/B equivalence, vowel clarity",
            },
            {
                "text": "El perro corre rápido por el jardín.",
                "focus": "Trilled RR, soft J sound",
            },
        ],
        "intermediate": [
            {
                "text": "Aunque llovía mucho, decidimos salir a caminar por la ciudad.",
                "focus": "LL sound, past tense pronunciation",
            },
            {
                "text": "El desarrollo tecnológico ha transformado nuestra manera de comunicarnos.",
                "focus": "Complex syllable stress, -ción ending",
            },
            {
                "text": "¿Podrías recomendarme un buen restaurante por aquí cerca?",
                "focus": "Conditional tense flow, soft R",
            },
            {
                "text": "La investigación científica requiere paciencia y dedicación constante.",
                "focus": "Academic register, -ción vs -ncia",
            },
            {
                "text": "Ayer fuimos al mercado y compramos frutas frescas y verduras de temporada.",
                "focus": "Past tense, natural rhythm and linking",
            },
        ],
    },
}

# Voice configuration for Google Cloud TTS Chirp 3 HD
VOICE_CONFIG = {
    "male": {
        "en-US": "en-US-Chirp3-HD-Algenib",
        "es-ES": "es-ES-Chirp3-HD-Algenib",
    },
    "female": {
        "en-US": "en-US-Chirp3-HD-Achernar",
        "es-ES": "es-ES-Chirp3-HD-Achernar",
    },
}


def get_sentences(language_code, difficulty=None):
    """Get sentences for a given language."""
    lang_data = SENTENCES.get(language_code, {})
    if difficulty:
        return lang_data.get(difficulty, [])
    # Return all difficulties combined
    all_sentences = []
    for diff in ["beginner", "intermediate"]:
        all_sentences.extend(lang_data.get(diff, []))
    return all_sentences


def get_voice_name(gender, language_code):
    """Get the Google TTS voice name for gender and language."""
    return VOICE_CONFIG.get(gender, {}).get(language_code)
