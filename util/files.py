import streamlit as st

def get_allowed_types():
    allowed_types = set(["png", "jpg", "jpeg", "jfif", "bmp"])
    return allowed_types


def allowed_file(filename):
    ALLOWED_EXTENSIONS = get_allowed_types()
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS