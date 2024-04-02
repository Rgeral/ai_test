#!/usr/bin/env python3

from ctransformers import AutoModelForCausalLM

def main():


    gpu_layers = 50

    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    model_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    model_type = "mistral"
    llm = AutoModelForCausalLM.from_pretrained(model_name, model_file=model_file, model_type=model_type, gpu_layers=gpu_layers)

    input_text = "can i give you some PDF to make you learn more things ?"
    generated_text = llm(input_text)
    print(generated_text)

if __name__ == "__main__":
    main()
 