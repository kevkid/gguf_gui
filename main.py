import streamlit as st
import sys
import os
import logging
from pathlib import Path
import numpy as np
import argparse
import llama_cpp
from llama_cpp import llama_model_quantize_params
from argparse import Namespace
import subprocess

# Import the main function from the convert module
sys.path.append(os.path.join(os.getcwd(), "llama.cpp"))
from convert import main

DEFAULT_CONCURRENCY = 2
script_path = "llama.cpp/convert-hf-to-gguf.py"
# Define the enum
ggml_type_enum = {
    "LLAMA_FTYPE_ALL_F32": 0,
    "LLAMA_FTYPE_MOSTLY_F16": 1,
    "LLAMA_FTYPE_MOSTLY_Q4_0": 2,
    "LLAMA_FTYPE_MOSTLY_Q4_1": 3,
    "LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16": 4,
    "LLAMA_FTYPE_MOSTLY_Q8_0": 7,
    "LLAMA_FTYPE_MOSTLY_Q5_0": 8,
    "LLAMA_FTYPE_MOSTLY_Q5_1": 9,
    "LLAMA_FTYPE_MOSTLY_Q2_K": 10,
    "LLAMA_FTYPE_MOSTLY_Q3_K_S": 11,
    "LLAMA_FTYPE_MOSTLY_Q3_K_M": 12,
    "LLAMA_FTYPE_MOSTLY_Q3_K_L": 13,
    "LLAMA_FTYPE_MOSTLY_Q4_K_S": 14,
    "LLAMA_FTYPE_MOSTLY_Q4_K_M": 15,
    "LLAMA_FTYPE_MOSTLY_Q5_K_S": 16,
    "LLAMA_FTYPE_MOSTLY_Q5_K_M": 17,
    "LLAMA_FTYPE_MOSTLY_Q6_K": 18,
    # "LLAMA_FTYPE_MOSTLY_IQ2_XXS": 19,
    # "LLAMA_FTYPE_MOSTLY_IQ2_XS": 20,
    # "LLAMA_FTYPE_MOSTLY_Q2_K_S": 21,
    # "LLAMA_FTYPE_MOSTLY_IQ3_XS": 22,
    # "LLAMA_FTYPE_MOSTLY_IQ3_XXS": 23,
    # "LLAMA_FTYPE_MOSTLY_IQ1_S": 24,
    # "LLAMA_FTYPE_MOSTLY_IQ4_NL": 25,
    # "LLAMA_FTYPE_MOSTLY_IQ3_S": 26,
    # "LLAMA_FTYPE_MOSTLY_IQ3_M": 27,
    # "LLAMA_FTYPE_MOSTLY_IQ2_S": 28,
    # "LLAMA_FTYPE_MOSTLY_IQ2_M": 29,
    # "LLAMA_FTYPE_MOSTLY_IQ4_XS": 30,
    # "LLAMA_FTYPE_MOSTLY_IQ1_M": 31,
    # "LLAMA_FTYPE_MOSTLY_BF16": 32
}

ggml_type_enum_invert = {v: k for k, v in ggml_type_enum.items()}


def streamlit_main():
    st.title("Model Conversion from Safetensor to GGUF")

    output_choices = (
        ["f32", "f16", "bf16", "q8_0", "auto"]
        if np.uint32(1) == np.uint32(1).newbyteorder("<")
        else ["f32", "f16"]
    )
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Select Directory")

    # Create a text input widget for manual entry
    manual_entry = st.text_input("Or Enter Directory Path Manually")
    outtype = "0"
    # outfile = ""
    # Use the selected directory or the manually entered directory
    if uploaded_file is not None:
        file_directory = os.path.dirname(uploaded_file.name)
        file_name = os.path.basename(file_directory)

        model = f"{file_directory}"
        outfile = f"{file_directory}/{file_name}_{outtype}.gguf"
    elif manual_entry:
        file_directory = os.path.dirname(manual_entry)
        file_name = os.path.basename(file_directory)
        model = f"{file_directory}"
        outfile = f"{file_directory}/{file_name}_{outtype}.gguf"
    else:
        st.warning("Please select a directory or enter a directory path.")

    # Display the selected or entered directory
    if uploaded_file is not None or manual_entry:
        st.write("Selected Directory:", file_directory)
    # Streamlit widgets to input arguments
    # outtype = st.selectbox("Output format", output_choices, index=0)
    outtype = st.selectbox("Output format", output_choices, index=0)
    awq_path = st.text_input("AWQ path (optional)")
    verbose = st.checkbox("Verbose logging")
    # outfile = st.text_input("Output file name (optional)")
    vocab_only = st.checkbox("Extract only the vocab")
    big_endian = st.checkbox("Model is executed on big endian machine")
    # Button to trigger the main function
    if st.button("Run Conversion"):
        try:
            with st.spinner(f"Converting Safetensors to {outtype}"):
                # Define the arguments you want to pass
                # For example:
                outfile = f"{file_directory}/{file_name}_{outtype}.gguf"
                if vocab_only:
                    outfile = outfile.replace(".gguf", "_vocab_only.gguf")
                if big_endian:
                    outfile = outfile.replace(".gguf", "_big_endian.gguf")
                st.success(f"Writing file to: {outfile}")
                args = [
                    "--outfile",
                    outfile,
                    "--outtype",
                    outtype,
                    "--verbose" if verbose else "",
                    "--vocab-only" if vocab_only else "",
                    "--bigendian" if big_endian else "",
                    model,
                ]
                args = [arg for arg in args if arg]  # Remove empty arguments
                # Call the main function with the arguments
                # Execute the script
                subprocess.run(["python", script_path] + args)
            st.success("Conversion completed successfully!")
            st.success("Model output filename:", outfile)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    # Display header
    st.header("Quantize FP32/16 GGUF")
    # Create a dropdown selector
    ggml_selected_type = st.selectbox("Select a GGML Type", list(ggml_type_enum.keys()))
    # Create a text input for the outfile
    outfile = st.text_input("Enter Output File Name", "")

    # Check if the "Quantize" button is clicked
    if st.button("Quantize"):
        # Check if outfile is provided
        if outfile:
            with st.spinner(f"Converting Safetensors to {ggml_selected_type}"):
                # Perform quantization
                output_fpath = (
                    f'{outfile.replace(".gguf", f"_{ggml_selected_type}")}.gguf'
                )
                result = llama_cpp.llama_model_quantize(
                    outfile.encode("utf-8"),
                    output_fpath.encode("utf-8"),
                    llama_model_quantize_params(
                        0, ggml_type_enum[ggml_selected_type], True, True, False
                    ),
                )
                st.success(f"""Quantizaion completed successfully!""")
                st.success(f"""Output quantized model to: {output_fpath}""")

        else:
            st.warning("Please enter an output file name.")


if __name__ == "__main__":
    streamlit_main()
