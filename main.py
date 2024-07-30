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
from huggingface_hub import snapshot_download
from pathlib import Path
# # Import the main function from the convert module
# sys.path.append(os.path.join(os.getcwd(), "llama.cpp"))
# from convert import main

# set the python path to include the llama.cpp directory
sys.path.append(os.path.join(os.getcwd(), "llama.cpp"))

uploaded_file = None

DEFAULT_CONCURRENCY = int(os.getenv("DEFAULT_CONCURRENCY", 2))

script_path = "llama.cpp/convert_hf_to_gguf.py"
# Define the enum
ggml_type_enum = {
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 8,
    "Q5_1": 9,
    "IQ2_XXS": 19,
    "IQ2_XS": 20,
    "IQ2_S": 28,
    "IQ2_M": 29,
    "IQ1_S": 24,
    "IQ1_M": 31,
    "Q2_K": 10,
    "Q2_K_S": 21,
    "IQ3_XXS": 23,
    "IQ3_S": 26,
    "IQ3_M": 27,
    "Q3_K": 12,  # alias for Q3_K_M
    "IQ3_XS": 22,
    "Q3_K_S": 11,
    "Q3_K_M": 12,
    "Q3_K_L": 13,
    "IQ4_NL": 25,
    "IQ4_XS": 30,
    "Q4_K": 15,  # alias for Q4_K_M
    "Q4_K_S": 14,
    "Q4_K_M": 15,
    "Q5_K": 17,  # alias for Q5_K_M
    "Q5_K_S": 16,
    "Q5_K_M": 17,
    "Q6_K": 18,
    "Q8_0": 7,
    "F16": 1,
    "BF16": 32,
    "F32": 0,
    "COPY": -1  # Special case for copying tensors without quantizing
}

ggml_type_enum_invert = {v: k for k, v in ggml_type_enum.items()}


def streamlit_main():
    st.title("GGUF_GUI")
    st.header("Step 1: Convert Safetensor to GGUF")
    output_choices = (
        ["f32", "f16", "bf16", "q8_0", "auto"]
        if np.uint32(1) == np.uint32(1).newbyteorder("<")
        else ["f32", "f16"]
    )
    # Create a file uploader widget
    # uploaded_file = st.file_uploader("Select Directory")

    # Create a text input widget for manual entry
    manual_entry = Path(st.text_input("Enter Directory Path or repo", placeholder="/path/to/safetensors/ or username_or_org/repo_name"))
    root_output_path = Path(st.text_input("Enter path to save your work to.", placeholder="/path/to/files/"))
    outtype = "0"
    outfile = ""
    # Streamlit widgets to input arguments
    outtype = st.selectbox("Output format", output_choices, index=0)
    awq_path = st.text_input("AWQ path (optional)")
    verbose = st.checkbox("Verbose logging")
    outfile_input = st.text_input("Output file name (optional)", "")
    vocab_only = st.checkbox("Extract only the vocab")
    big_endian = st.checkbox("Model is executed on big endian machine")
    # Button to trigger the main function
    if st.button("Run Conversion"):
        try:
            with st.spinner(f"Converting Safetensors to {outtype}"):
                # Define the arguments you want to pass
                try:
                    safetensor_dl_loc = snapshot_download(repo_id=str(manual_entry), local_files_only=False, local_dir=str(root_output_path.joinpath(manual_entry.name)))
                    st.success(f"Model downloaded to {safetensor_dl_loc}")
                except Exception as e:
                    print(e)
                outfile = f"{str(root_output_path)}/{manual_entry.name}_{outtype}.gguf"
                if vocab_only:
                    outfile = outfile.replace(".gguf", "_vocab_only.gguf")
                if big_endian:
                    outfile = outfile.replace(".gguf", "_big_endian.gguf")
                st_to_gguf_output = f"{root_output_path.joinpath(manual_entry.name)}.gguf" or outfile
                st.session_state['st_to_gguf_outfile'] = st_to_gguf_output

                args = [
                    "--outfile",
                    st_to_gguf_output,
                    "--outtype",
                    outtype,
                    "--verbose" if verbose else "",
                    "--vocab-only" if vocab_only else "",
                    "--bigendian" if big_endian else "",
                    safetensor_dl_loc,
                ]
                args = [arg for arg in args if arg]  # Remove empty arguments
                # Call the main function with the arguments
                # Execute the script
                subprocess.run(["python", script_path] + args)
                st.success("Conversion completed successfully!")
                st.success(f"Wrote file to: {st_to_gguf_output}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    # Display header
    st.header("Step 2: Quantize FP32/16")
    # Use quantize: ./quantize
    # Create a dropdown selector
    ggml_selected_type = st.selectbox("Select a Quantization Type", list(ggml_type_enum.keys()))
    # Create a text input for the outfile
    outfile_ggml = st.text_input("Enter file path to your FP 32/16 GGUF", st.session_state.get('st_to_gguf_outfile'), placeholder="/path/to/input/model.gguf")
    # Optional options
    with st.expander("Optional Parameters", expanded=False):
        model_quant = st.text_input("Quantized Model File Path (model-quant.gguf, optional)", "")
        allow_requantize = st.checkbox("Allow Requantize (--allow-requantize)")
        leave_output_tensor = st.checkbox("Leave Output Tensor (--leave-output-tensor)")
        pure = st.checkbox("Pure (--pure)")
        imatrix_file = st.text_input("Importance Matrix File (--imatrix)", "")
        include_weights = st.text_input("Include Weights Tensor (--include-weights)", "")
        exclude_weights = st.text_input("Exclude Weights Tensor (--exclude-weights)", "")
        output_tensor_type = st.text_input("Output Tensor Type (--output-tensor-type)", "")
        token_embedding_type = st.text_input("Token Embedding Type (--token-embedding-type)", "")
        override_kv = st.text_area("Override KV (--override-kv)", "")
        nthreads = st.number_input("Number of Threads (nthreads)", min_value=1, step=1)

    # Ensure --include-weights and --exclude-weights are not used together
    if include_weights and exclude_weights:
        st.error("Cannot use --include-weights and --exclude-weights together.")
        return
    # Use iMatrix: ./imatrix
    # Checkbox to execute the command
    imatrix = st.checkbox("iMatrix")

    # Execute the command if there is an imatrix is checked
    if imatrix:  # for the UI
        # Optional options
        with st.expander("iMatrix Parameters", expanded=True):
            # Mandatory options
            training_data = st.text_input("Training Data File Path (-f)", "")
            imatrix_output_file = st.text_input("Output File Path (-o)", root_output_path.joinpath("imatrix.dat"))
            verbosity_level = st.selectbox("Verbosity Level (--verbosity)", [None, "0", "1", "2", "3"], index=0)
            num_chunks = st.number_input("Number of Chunks (-ofreq)", min_value=1, step=1)
            ow_option = st.selectbox("Overwrite Option (-ow)", [None, "0", "1"], index=0)

            # Add any other common params here
            other_params = st.text_area("Other Parameters", "", placeholder="--arg value --arg2 value2")

    # Check if the "Quantize" button is clicked
    if st.button("Quantize"):
        # Check if outfile is provided
        if outfile_ggml:
            with st.spinner(f"Converting Safetensors to {ggml_selected_type}"):
                if imatrix:
                    # Construct the command
                    cmd = ["./llama.cpp/imatrix"]
                    if outfile_ggml:
                        cmd.extend(["-m", outfile_ggml])
                    if training_data:
                        cmd.extend(["-f", training_data])
                    if imatrix_output_file:
                        cmd.extend(["-o", imatrix_output_file])
                    if verbosity_level:
                        cmd.extend(["--verbosity", verbosity_level])
                    if num_chunks:
                        cmd.extend(["-ofreq", str(num_chunks)])
                    if ow_option:
                        cmd.extend(["-ow", ow_option])
                    if other_params:
                        cmd.extend(other_params.split())
                    print("#####################imatrix############################")
                    result = subprocess.run(cmd, capture_output=False, text=False)
                    # st.text_area("Command Output", result.stdout)
                    # st.text_area("Command Errors", result.stderr)

                # Construct the command
                cmd = ["./llama.cpp/quantize"]
                if imatrix:
                    cmd.extend(["--imatrix", imatrix_output_file or imatrix_file])
                if allow_requantize:
                    cmd.append("--allow-requantize")
                if leave_output_tensor:
                    cmd.append("--leave-output-tensor")
                if pure:
                    cmd.append("--pure")
                if include_weights:
                    cmd.extend(["--include-weights", include_weights])
                if exclude_weights:
                    cmd.extend(["--exclude-weights", exclude_weights])
                if output_tensor_type:
                    cmd.extend(["--output-tensor-type", output_tensor_type])
                if token_embedding_type:
                    cmd.extend(["--token-embedding-type", token_embedding_type])
                if override_kv:
                    for kv in override_kv.split('\n'):
                        if kv.strip():
                            cmd.extend(["--override-kv", kv.strip()])
                infile_ggml = (outfile_ggml or st.session_state.get('st_to_gguf_outfile'))
                outfile_ggml = infile_ggml.replace(".gguf", f"_{ggml_selected_type}.gguf")

                if infile_ggml:
                    cmd.append(infile_ggml)
                if outfile_ggml:
                    cmd.append(outfile_ggml)
                if model_quant:
                    cmd.append(model_quant)
                cmd.append(str(ggml_type_enum[ggml_selected_type]))
                if nthreads:
                    cmd.append(str(nthreads))
                print("#####################Quantize############################")
                # Quantize the model
                result = subprocess.run(cmd, capture_output=False, text=False)
                # st.text_area("Command Output", result)
                # st.text_area("Command Errors", result.stderr)

                # ######
                st.success(f"""Quantizaion completed successfully!""")
                st.success(f"""Output quantized model to: {outfile_ggml}""")

        else:
            st.warning("Please enter an output file name.")


if __name__ == "__main__":
    streamlit_main()
