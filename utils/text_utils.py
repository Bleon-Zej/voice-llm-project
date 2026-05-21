# utils/text_utils.py


def is_sentence(buffer: str, next_token: str) -> bool:
    return buffer.endswith((".", "?", "!")) and next_token == " "
