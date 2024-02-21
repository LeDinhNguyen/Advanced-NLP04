
import time
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import accelerate


if __name__ == '__main__':
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

    # STEP_1: Init an empty skeleton of the model which won’t take up any RAM
    config = AutoConfig.from_pretrained(model_name)
    with accelerate.init_empty_weights():
        dummy_model = AutoModelForCausalLM.from_config(config)
    
    # let Accelerate automatically handle it or design device_map yourself 
    # - device_map="auto" ("balanced", "balanced_low_0", "sequential") (GPU > CPU > disk)
    # - device_map = {"block1": 0, "block2": 1}
    # - max_memory={0: "4GiB", 1: "8GiB", "cpu": "5GiB", "disk": "30GiB"}
    # device_map = accelerate.infer_auto_device_map(dummy_model, max_memory={0: "40GiB", "cpu": "50GiB"})
    device_map = accelerate.infer_auto_device_map(dummy_model, max_memory={"cpu": "4GiB"})

    # STEP_2: Load model with device_map & offload_state_dict=True
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        offload_state_dict=True, # offload the CPU state dict to disk to avoid getting out of CPU RAM
        offload_folder="offload"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # STEP_3: Generate output
    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        # {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    # device = "cuda"
    device = "cpu"
    model_inputs = encodeds.to(device)

    start = time.time()
    generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
    end = time.time()
    print(f'====Inference time====: {end-start}')