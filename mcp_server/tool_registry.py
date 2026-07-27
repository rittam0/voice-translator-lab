TOOLS = [
    {
        "name": "translate_voice",
        "description": (
            "Run local ASR, English-Hindi translation, and voice-cloned TTS "
            "for one complete audio clip."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_base64": {"type": "string"},
                "source_language": {"type": "string", "enum": ["en", "hi"]},
                "target_language": {"type": "string", "enum": ["en", "hi"]},
            },
            "required": [
                "audio_base64",
                "source_language",
                "target_language",
            ],
        },
    }
]
