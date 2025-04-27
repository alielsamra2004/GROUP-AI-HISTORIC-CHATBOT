# Interactive Historian: Colorize & Chat

This project provides a Gradio web application that allows users to:
1.  Upload a grayscale historical image.
2.  Colorize the image using a pre-trained deep learning model (SIGGRAPH 17).
3.  Chat with an AI historian (powered by Google Gemini) about the context of the image.

## Project Structure

```
📦 GROUP-AI/
┣ 📂 colorizers/
┃ ┣ 📄 __init__.py       # Makes 'colorizers' a Python package.
┃ ┣ 📄 base_color.py     # Base class definition for colorization models.
┃ ┣ 📄 eccv16.py         # Implementation of the ECCV16 colorization model.
┃ ┣ 📄 siggraph17.py     # Implementation of the SIGGRAPH17 colorization model (used by the app).
┃ ┗ 📄 util.py           # Utility functions for image loading, resizing, pre/post-processing.
┣ 📂 imgs/                # Directory containing sample input images.
┣ 📂 .venv/               # Virtual environment directory (auto-generated).
┣ 📄 historian_chatbot.py # The main Gradio application script. Handles UI, colorization, and Gemini API interaction.
┣ 📄 demo_release.py     # Original standalone demo script for colorization (not used by the main app).
┣ 📄 requirements.txt    # Lists required Python packages.
┣ 📄 README.md           # This file.
┗ 📄 LICENSE             # Project license file.
```
*(Note: `__pycache__`, `.DS_Store`, saved images like `saved_*.png`, etc. are omitted for clarity)*

## File Explanations

*   **`historian_chatbot.py`**: This is the core script that launches the Gradio web interface. It handles:
    *   Loading the SIGGRAPH17 colorization model.
    *   Setting up the Gradio UI using `gr.Blocks`.
    *   Processing image uploads and triggering colorization.
    *   Interacting with the Google Gemini API to provide historical context based on user queries and optional image descriptions.
*   **`colorizers/`**: This directory contains the code related to the image colorization models.
    *   `__init__.py`: Standard Python file to treat the directory as a package. Imports modules.
    *   `base_color.py`: Defines a base PyTorch `nn.Module` class from which specific colorizer models inherit.
    *   `eccv16.py`: Defines the neural network architecture and loads weights for the colorization model presented at ECCV 2016.
    *   `siggraph17.py`: Defines the architecture and loads weights for the improved colorization model presented at SIGGRAPH 2017. This is the model currently used by `historian_chatbot.py`.
    *   `util.py`: Provides helper functions for image handling: loading images (`load_img`), resizing (`resize_img`), converting between color spaces and preparing tensors for the models (`preprocess_img`), and converting model output back to a viewable image (`postprocess_tens`).
*   **`imgs/`**: Contains example grayscale images that can be used with the application.
*   **`demo_release.py`**: A standalone Python script (likely the original demo for the colorization code) that runs colorization on a single image specified via command-line arguments and saves/displays the output using Matplotlib. It is not directly used by the Gradio app.
*   **`requirements.txt`**: Specifies the Python libraries needed to run the project.
*   **`README.md`**: Provides information about the project, its structure, and how to run it.
*   **`LICENSE`**: Contains the license information for the project code.

## How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd GROUP-AI # Or your project directory name
    ```

2.  **Create and Activate a Virtual Environment:** (Recommended)
    ```bash
    python -m venv .venv  # Or python3 -m venv .venv
    # Activate the environment:
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    # .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install Gradio, Google Generative AI, PyTorch, scikit-image, and other necessary libraries. The colorizer models might download pre-trained weights on first run.

4.  **Set Up Google Gemini API Key:**
    *   You need an API key from Google AI Studio (formerly MakerSuite).
    *   Open the `historian_chatbot.py` script.
    *   Find the line `API_KEY = "YOUR-API-KEY-HERE"` (or similar).
    *   Replace `"YOUR-API-KEY-HERE"` with your actual Gemini API key.
    *   **Security Note:** For better security, consider setting the API key as an environment variable (e.g., `GEMINI_API_KEY`) and loading it in the script using `os.getenv("GEMINI_API_KEY")` instead of hardcoding it.

5.  **Run the Gradio App:**
    ```bash
    python historian_chatbot.py
    ```

6.  **Access the App:**
    *   Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860` or similar).
    *   Upload a grayscale image, click "Colorize Image".
    *   Optionally add text context about the image.
    *   Ask the historian chatbot questions in the chat interface.

## Citation

The colorization code used in this project is based on the following works. If you find these models useful for your research, please consider citing them:

```bibtex
@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={9},
  number={4},
  year={2017},
  publisher={ACM}
}
```

