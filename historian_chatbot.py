import gradio as gr
import google.generativeai as genai
import os
import sys # Add sys import

# Add the parent directory to the Python path to find 'colorizers'
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '.')) # Get current dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from PIL import Image

# --- Import Colorization Code ---
# Assuming 'colorizers' directory is in the same path or Python path
# Removed try...except ImportError to get a more specific error if import fails
from colorizers import siggraph17 # Use SIGGRAPH17 model
from colorizers import util

# Define placeholder variables in case the above imports *do* fail,
# although Python should raise an error before this point now.
# This prevents NameErrors later if the script somehow continues.
if 'siggraph17' not in locals(): siggraph17 = None
if 'util' not in locals(): util = None

# --- Configuration ---
API_KEY = "AIzaSyAtA_a0yFyB6he2EtHyl2fad_8slktf4aw" # Replace with your key or use env vars

# --- Initialize APIs and Models ---
# Gemini API
gemini_model = None
try:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Gradio app will still launch, but chatbot functionality will be limited.

# Colorization Model (Load on CPU)
colorizer_model = None
if siggraph17: # Check if import was successful
    try:
        colorizer_model = siggraph17(pretrained=True).eval()
        # Note: No .cuda() call, model stays on CPU
        print("Colorizer model loaded successfully (CPU).")
    except Exception as e:
        print(f"Error loading colorizer model: {e}")
else:
     print("Colorizer module not available.")


# --- Historian Persona ---
HISTORIAN_PROMPT = (
    "You are a distinguished historian AI engaging in a conversation. "
    "When asked about a historical figure, event, or artwork, provide a brief, formal introduction (2-3 sentences maximum) touching upon its significance. "
    "Then, to encourage further discussion, suggest 2-3 specific follow-up questions the user might be interested in asking. Frame these as suggestions, for example: 'Perhaps you'd like to delve into their early influences, their most significant work, or their lasting legacy?'"
    "If provided with context about a specific image, subtly incorporate it into your brief introduction. "
    "Maintain a scholarly and objective tone."
)

# --- Core Logic Functions ---

def colorize_image(input_image_np):
    """
    Colorizes an input NumPy array image using the loaded SIGGRAPH17 model.
    Expects input_image_np to be a NumPy array (H, W, 3) from Gradio Image input.
    Returns the colorized image as a NumPy array (H, W, 3).
    """
    print(f"DEBUG: colorize_image called. Input shape: {input_image_np.shape if input_image_np is not None else 'None'}, Input dtype: {input_image_np.dtype if input_image_np is not None else 'None'}") # DEBUG

    if colorizer_model is None or util is None:
        print("Colorizer model or util not available.")
        return input_image_np # Or create a dummy error image

    if input_image_np is None:
        return None 

    try:
        # 1. Preprocess the image (using functions from colorizers.util)
        if input_image_np.ndim == 2:
            img_rgb_orig = np.tile(input_image_np[:, :, None], 3)
        elif input_image_np.shape[2] == 1:
             img_rgb_orig = np.tile(input_image_np, 3)
        else:
            img_rgb_orig = input_image_np

        # Pass the uint8 image directly
        print(f"DEBUG: Calling preprocess_img with shape: {img_rgb_orig.shape}, dtype: {img_rgb_orig.dtype}") # DEBUG

        (tens_l_orig, tens_l_rs) = util.preprocess_img(img_rgb_orig, HW=(256, 256))
        print(f"DEBUG: tens_l_orig shape: {tens_l_orig.shape}, dtype: {tens_l_orig.dtype}") # DEBUG
        print(f"DEBUG: tens_l_rs shape: {tens_l_rs.shape}, dtype: {tens_l_rs.dtype}") # DEBUG
        
        # 2. Run the colorizer model (on CPU)
        with torch.no_grad():
             out_ab = colorizer_model(tens_l_rs)
        print(f"DEBUG: Model output out_ab shape: {out_ab.shape}, dtype: {out_ab.dtype}") # DEBUG

        # 3. Postprocess to get the final RGB image
        colorized_img_np = util.postprocess_tens(tens_l_orig, out_ab)
        print(f"DEBUG: Postprocessed colorized_img_np shape: {colorized_img_np.shape}, dtype: {colorized_img_np.dtype}") # DEBUG

        # Convert from float 0-1 back to uint8 0-255 for display
        colorized_img_np_uint8 = (colorized_img_np * 255).astype(np.uint8)

        print("Image colorized successfully.")
        return colorized_img_np_uint8

    except Exception as e:
        print(f"Error during colorization: {e}")
        # Return original image or an error indicator in case of failure
        # Converting original numpy array to displayable format if needed
        if input_image_np.max() <= 1.0: # If it was normalized
            return (input_image_np * 255).astype(np.uint8) 
        else:
             return input_image_np.astype(np.uint8)


def get_historian_response(chat_history, user_question, image_context):
    """
    Gets a response from the Gemini API, formats it for Gradio chat.
    Uses the existing chat history, new question, and optional image context.
    """
    if not gemini_model:
         # Append error message to chat history
        chat_history.append((user_question, "Sorry, the AI historian model is not available."))
        return chat_history, "" # Return updated history and clear textbox

    if not user_question:
        # Optionally handle empty input, maybe add a message to chat
        # chat_history.append((None, "Please enter a question.")) 
        return chat_history, "" # Keep history, clear textbox

    # Construct the prompt for the Gemini model
    prompt_parts = [HISTORIAN_PROMPT]
    if image_context:
        prompt_parts.append("\n--- Provided Image Context ---")
        prompt_parts.append(image_context)
        prompt_parts.append("--- End Image Context ---")
    
    # Add user's current question
    prompt_parts.append("\n--- User Query ---")
    prompt_parts.append(user_question)
    prompt_parts.append("--- End User Query ---")

    full_prompt = "\n".join(prompt_parts)
    
    print("\n--- Sending Prompt to Gemini ---")
    # print(full_prompt) # Uncomment for debugging
    print("--- End Prompt ---")


    try:
        # Generate the response using the Gemini API
        response = gemini_model.generate_content(full_prompt)
        
        if response.parts:
            historian_answer = response.text
        else:
            # Handle blocked/empty responses
            safety_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else "No feedback available."
            historian_answer = f"Sorry, I couldn't generate a response. Possible safety block or empty response. Feedback: {safety_feedback}"

    except Exception as e:
        print(f"Gemini API Error: {e}")
        historian_answer = "Sorry, something went wrong with the AI historian service. Please try again later."

    # Append user question and AI response to chat history
    chat_history.append((user_question, historian_answer))

    # Return updated history and clear the input textbox
    return chat_history, ""


# --- Gradio Interface Definition using Blocks ---
with gr.Blocks(theme="soft") as app:
    gr.Markdown("# Historical Image Colorizer & Chatbot")
    gr.Markdown("Upload a grayscale historical image, see it colorized, then ask the AI historian about it.")

    with gr.Row():
        # Image Column
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Grayscale Image", sources=["upload", "clipboard"])
            colorize_button = gr.Button("Colorize Image")
            image_output = gr.Image(type="numpy", label="Colorized Image")
            
            gr.Markdown("_(Optional) Add context about the image below before asking your question._")
            image_context_input = gr.Textbox(
                label="Image Context", 
                placeholder="e.g., 'Painting of Napoleon's coronation', 'Photo of the Pyramids of Giza'",
                lines=3
            )

        # Chat Column
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="Historian Chat", 
                bubble_full_width=False,
                height=600,
                # type='messages' # This seems unnecessary with gr.Chatbot within gr.Blocks
                )
            question_input = gr.Textbox(label="Ask the Historian", placeholder="Type your question about the image or topic here...")
            submit_button = gr.Button("Ask Question")


    # --- Component Interactions ---
    
    # 1. Colorize Button Action
    colorize_button.click(
        fn=colorize_image,
        inputs=[image_input],
        outputs=[image_output]
    )

    # 2. Submit Question Action (using chatbot pattern)
    # We need a function that takes history, question, context and returns updated history + cleared textbox
    submit_button.click(
         fn=get_historian_response,
         inputs=[chatbot, question_input, image_context_input],
         outputs=[chatbot, question_input] # Update chatbot history, clear question textbox
    )
    
    # Allow submitting question with Enter key
    question_input.submit(
         fn=get_historian_response,
         inputs=[chatbot, question_input, image_context_input],
         outputs=[chatbot, question_input] 
    )

    # Add examples (might need adjustment for Blocks layout)
    # gr.Examples( # Examples are a bit trickier with multi-input functions in Blocks
    #     examples=[...], 
    #     inputs=[question_input, image_context_input] # Specify which inputs the examples map to
    # )


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    if API_KEY == "YOUR-API-KEY-HERE" or not API_KEY: # Also check if empty
        print("\n*** WARNING: Gemini API key is not set. Please set the API_KEY variable in the script. ***\n")
    if colorizer_model is None:
         print("\n*** WARNING: Colorizer model could not be loaded. Colorization will not work. ***\n")
    if gemini_model is None:
         print("\n*** WARNING: Gemini model could not be loaded. Chatbot functionality will be limited. ***\n")

    app.launch()
    print("Gradio App stopped.")