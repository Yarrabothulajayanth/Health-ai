import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Model details
model_name = "ibm-granite/granite-3.3-2b-instruct"
hf_token = os.getenv("HF_TOKEN")  # Secure token from Hugging Face secrets

# Load model & tokenizer using token
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", use_auth_token=hf_token)

# Generate response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interface functions
def patient_chat_interface(question):
    return generate_response(f"You are a health assistant. Answer this:\n{question}")

def predict_disease_interface(symptoms):
    return generate_response(f"Based on these symptoms, what diseases might this person have?\n{symptoms}")

def treatment_plan_interface(condition):
    return generate_response(f"Suggest a treatment plan for this condition:\n{condition}")

# Gradio UI
with gr.Blocks() as demo:
    gr.HTML("""
        <style>
            body { background-color: #d4fcd6 !important; }
            .gradio-container { background-color: #d4fcd6 !important; }
        </style>
    """)

    gr.Markdown("## 🧠 Healthy Intelligent Health Care Assistant")

    with gr.Tab("🩺 Patient Chat"):
        question = gr.Textbox(label="Ask a health question")
        response = gr.Textbox(label="AI Response")
        gr.Button("Ask").click(patient_chat_interface, inputs=question, outputs=response)

    with gr.Tab("🧬 Disease Prediction"):
        symptoms = gr.Textbox(label="Enter your symptoms")
        prediction = gr.Textbox(label="Predicted Diseases")
        gr.Button("Predict").click(predict_disease_interface, inputs=symptoms, outputs=prediction)

    with gr.Tab("💊 Treatment Plan"):
        condition = gr.Textbox(label="Enter the diagnosed disease/condition")
        treatment = gr.Textbox(label="Suggested Treatment Plan")
        gr.Button("Get Plan").click(treatment_plan_interface, inputs=condition, outputs=treatment)

demo.launch()
