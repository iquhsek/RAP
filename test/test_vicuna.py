"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse
import torch
from fastchat.model import load_model, get_conversation_template, add_model_args
import colorama
from colorama import Fore
from colorama import Style
colorama.init()


@torch.inference_mode()
def main(args):
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    # TODO:
    print (f'|#|#|#|#|#|#|{Fore.CYAN}Info about the model: {Style.RESET_ALL}{Fore.RED}{model.__class__, model.__class__.__name__}{Style.RESET_ALL}')

    msg = args.message

    # TODO:
    print (f'|#|#|#|#|#|#|{Fore.CYAN}msg = {Style.RESET_ALL}{Fore.RED}{msg}{Style.RESET_ALL}')

    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # TODO:
    print (f'|#|#|#|#|#|#|{Fore.CYAN}PROMPT = {Style.RESET_ALL}{Fore.RED}{prompt}{Style.RESET_ALL}')

    input_ids = tokenizer([prompt]).input_ids
    
    # TODO:
    print (f'|#|#|#|#|#|#|{Fore.CYAN}input_ids = {Style.RESET_ALL}{Fore.RED}{input_ids}{Style.RESET_ALL}')
    
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    print(f"{conv.roles[0]}: {msg}")
    print(f"{conv.roles[1]}: {outputs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Tell me something about Evanston.")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)