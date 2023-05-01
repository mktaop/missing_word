#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:05:05 2023

@author: avi_patel
"""
import streamlit as st
from transformers import pipeline

# Load the BERT pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Define the Streamlit app
def app():
    # Set the app title
    st.title("BERT-based Word Completion")

    # Get user input
    sentence = st.text_input("Enter a sentence with a missing word (use 'mask' to indicate the missing word):", "")

    # Process the input with BERT
    if sentence:
        # Find the mask token in the input sentence
        mask_token_index = sentence.find("mask")
        if mask_token_index == -1:
            st.warning("Please include the word 'mask' in your input sentence.")
            return

        # Replace the mask token with the BERT mask token
        sentence = sentence.replace("mask", fill_mask.tokenizer.mask_token)

        # Get the BERT predictions
        predictions = fill_mask(sentence)

        # Display the top 5 predictions
        st.write("Top 5 Predictions:")
        for prediction in predictions[:5]:
            st.write(f"- {prediction['sequence']} (score: {prediction['score']:.4f})")


# Run the app
app()



