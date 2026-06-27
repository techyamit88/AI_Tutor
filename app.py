import os
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the Hugging Face Serverless client.
# Note: For production use, set up a free Hugging Face token at huggingface.co/settings/tokens
# and pass it as InferenceClient(token="your_token_here") or configure it as an environment variable.
HF_TOKEN = os.getenv("HF_TOKEN", "")
client = InferenceClient(token=HF_TOKEN)
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# ----------------------------------------------------
# Core Generation Functions (Using Streaming)
# ----------------------------------------------------

def build_lesson(topic, audience_level):
    """Generates structured notes and custom everyday analogies."""
    system_prompt = (
        "You are an expert AI Tutor. Your task is to break down advanced, dense topics into "
        "exceptionally clear, accessible explanations. Always use clean Markdown formatting with bold terms."
    )
    user_prompt = f"""
    Create a complete lesson module for a student at the **{audience_level}** level.
    
    Topic: {topic}
    
    Structure your output exactly like this:
    # 📚 Lesson Module: {topic}
    
    ## 🎯 Core Concepts Explained
    (Provide an intuitive, clear breakdown of what this is and why it matters in 2-3 short sections.)
    
    ## 💡 The Everyday Analogy
    (Create a highly relatable, real-world analogy. For instance, comparing tokenizers to cutting a sentence up into lego bricks, or neural networks to baking recipes. Make it stick!)
    
    ## 📌 Key Takeaways
    - Bullet point 1
    - Bullet point 2
    """
    
    response = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Use chat.completions.create with stream=True for real-time text output
    stream = client.chat.completions.create(model=MODEL_ID, messages=messages, stream=True, max_tokens=2000)
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            response += token
            yield response

def build_assessment(topic, audience_level, num_questions):
    """Generates customized daily quizzes, tests, and answer keys."""
    system_prompt = "You are an educational assessment designer. Your goal is to write unambiguous, highly educational exam items."
    user_prompt = f"""
    Generate a **{num_questions}-question daily assessment quiz** regarding the following topic: {topic}.
    Target Audience Level: {audience_level}
    
    Format requirements:
    - Mix Multiple Choice Questions (MCQs) and True/False questions.
    - Provide clear, conceptual options.
    - Put the complete **Answer Key & Explanations** at the very bottom of the document hidden cleanly under a header.
    """
    
    response = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    stream = client.chat.completions.create(model=MODEL_ID, messages=messages, stream=True, max_tokens=1500)
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            response += token
            yield response

def simplify_code(raw_code, programming_language):
    """Annotates dense machine learning/AI source code line-by-line for students."""
    system_prompt = "You are a senior computer science teacher who annotates complex source code to make it approachable for beginners."
    user_prompt = f"""
    Analyze and break down the following {programming_language} code block so an AI student can understand it.
    
    ```python
    {raw_code}
    ```
    
    Provide your output in two sections:
    1. **🚶 Line-by-Line Breakdown:** Paste back key code sections with detailed explanation bullets explaining what functions, hyper-parameters, or logic rules are executing.
    2. **🛠️ Interactive Micro-Challenge:** Write one quick task or adjustment the student should make to this code to test their practical understanding (e.g., 'Try altering the learning rate parameter or changing the padding rules...').
    """
    
    response = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    stream = client.chat.completions.create(model=MODEL_ID, messages=messages, stream=True, max_tokens=2000)
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            response += token
            yield response


# ----------------------------------------------------
# Gradio User Interface Layout Architecture
# ----------------------------------------------------

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate")) as app:
    gr.Markdown("# 🎓 AI Educator Engine & Tutor Copilot")
    gr.Markdown("Streamline curriculum design, generate instant targeted homework, and turn intricate code blocks into student lessons.")
    
    # Setup shared configurations used by multiple tabs
    levels = ["Absolute Beginner (No Technical Background)", "Intermediate (Basic Tech/Python Knowledge)", "Advanced (Data Scientists / Software Engineers)"]
    
    with gr.Tabs():
        
        # TAB 1: CORE LESSON PLANNING
        with gr.TabItem("📚 1. Lesson & Analogy Builder"):
            gr.Markdown("### Create Intuitive, High-Retention Lecture Notes")
            with gr.Row():
                with gr.Column(scale=1):
                    lesson_topic = gr.Textbox(label="Lesson Topic", placeholder="e.g., Vector Databases, Backpropagation, Attention Mechanism", lines=2)
                    lesson_level = gr.Dropdown(label="Target Student Level", choices=levels, value=levels[0])
                    generate_lesson_btn = gr.Button("Build Lecture Notes", variant="primary")
                with gr.Column(scale=2):
                    lesson_output = gr.Markdown(value="*Your structured lecture material will stream here...*")
            
            # Event Listener Wiring
            generate_lesson_btn.click(fn=build_lesson, inputs=[lesson_topic, lesson_level], outputs=lesson_output)
            
        # TAB 2: DAILY ASSESSMENTS
        with gr.TabItem("📝 2. Quiz & Exam Planner"):
            gr.Markdown("### Generate Homework, MCQs, and Conceptual Assessments")
            with gr.Row():
                with gr.Column(scale=1):
                    quiz_topic = gr.Textbox(label="Quiz Subject/Scope", placeholder="e.g., Prompt Engineering Basics, Basic Neural Network Architecture", lines=2)
                    quiz_level = gr.Dropdown(label="Difficulty Tier", choices=levels, value=levels[0])
                    quiz_count = gr.Slider(label="Number of Questions", minimum=2, maximum=10, step=1, value=5)
                    generate_quiz_btn = gr.Button("Generate Exam Papers", variant="primary")
                with gr.Column(scale=2):
                    quiz_output = gr.Markdown(value="*Your tailored exam questions and key will stream here...*")
                    
            generate_quiz_btn.click(fn=build_assessment, inputs=[quiz_topic, quiz_level, quiz_count], outputs=quiz_output)

        # TAB 3: CODE DECONSTRUCTION TOOL
        with gr.TabItem("💻 3. Code Explainer for Tutors"):
            gr.Markdown("### Convert Raw Scripts into Student-Friendly Technical Breakdown Labs")
            with gr.Row():
                with gr.Column(scale=1):
                    input_language = gr.Dropdown(label="Language / Framework", choices=["Python (PyTorch / Transformers)", "Python (Pandas / Scikit-Learn)", "JavaScript", "SQL / Database"], value="Python (PyTorch / Transformers)")
                    input_code = gr.Code(label="Paste Complex Code Block Here", language="python", lines=12, value="from transformers import pipeline\n\nclassifier = pipeline('sentiment-analysis')\nres = classifier('I love building tools!')\nprint(res)")
                    generate_code_btn = gr.Button("Deconstruct Code Script", variant="primary")
                with gr.Column(scale=2):
                    code_output = gr.Markdown(value="*Annotated educational code insights will stream here...*")
                    
            generate_code_btn.click(fn=simplify_code, inputs=[input_code, input_language], outputs=code_output)

# ----------------------------------------------------
# App Execution Check
# ----------------------------------------------------
if __name__ == "__main__":
    app.queue().launch()
