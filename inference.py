from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


# LLaMA class is used to encapsulate the model, tokenizer, and model args.
# The build method is a static method that is used to construct a new instance
# of the LLaMA class.

# The build method does the following:
# - If load_model is True, it loads the checkpoint from the checkpoints directory.
# - It reads the params.json file to get the model parameters.
# - It creates a new instance of ModelArgs with the parameters and the values
#   passed to the build method.
# - It loads the tokenizer.
# - It sets the default tensor type according to the device.
# - It creates a new instance of Transformer with the model args.
# - If load_model is True, it loads the checkpoint state dict into the model.


class LLaMA:

    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs,
    ):
        # The __init__ method initializes the LLaMA object.
        # It takes a model, tokenizer, and model args as parameters.
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        # The build method is a static method that is used to construct a new
        # instance of the LLaMA class.
        # It takes the following parameters:
        # - checkpoints_dir: the directory where the checkpoint files are located.
        # - tokenizer_path: the path to the tokenizer model.
        # - load_model: a boolean indicating whether to load the checkpoint.
        # - max_seq_len: the maximum sequence length.
        # - max_batch_size: the maximum batch size.
        # - device: the device to use.

        prev_time = time.time()
        # If load_model is True, it loads the checkpoint from the checkpoints directory.
        if load_model:

            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert (
                len(checkpoints) > 0
            ), f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()

        # It reads the params.json file to get the model parameters.
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # It creates a new instance of ModelArgs with the parameters and the values
        # passed to the build method.
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        # It loads the tokenizer.
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # It sets the default tensor type according to the device.
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        # It creates a new instance of Transformer with the model args.
        model = Transformer(model_args).to(device)

        # If load_model is True, it loads the checkpoint state dict into the model.
        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded State dict in {(time.time() - prev_time):.2f}")

        # It returns a new instance of the LLaMA class.
        return LLaMA(model, tokenizer, model_args)

    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        # The text_completion method is used to generate text completions.
        # It takes the following parameters:
        # - prompts: a list of prompts to generate completions for.
        # - temperature: the temperature to use for the generation.
        # - float: the float to use for the generation.
        # - top_p: the top_p to use for the generation.
        # - max_gen_len: the maximum length of the generated text.
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # convert each prompt into tokens
        prompt_tokens = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]

        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert (
            batch_size <= self.args.max_batch_size
        ), f"Batch size {batch_size} exceeds max batch size {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)

        # Make sure the prompt length is not too large
        assert (
            max_prompt_len <= self.args.max_seq_len
        ), f"Prompt length {max_prompt_len} exceeds max prompt length {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # create a list that will contain the generated tokens, along with initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=device
        )
        for k, t in enumerate(prompt_tokens):
            # populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = (
            tokens != pad_id
        )  # True if token is a prompt token, False otherwise

        # generate the text completions
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)

            else:
                # Greedily select the token with the max prob
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace the token if it is a padding token
            next_token = torch.where(
                prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id()
            )
            if all(eos_reached):
                break

        # decode the tokens into text
        out_tokens = []
        out_text = []

        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        "Simply, put the theory of relativity states that ",
        "If google was an Indian company founded in Mumbai, it would ",
        # Few short prompts
        """Translate English to French:
        sea otter => loutre de mer 
        peppermind => menthe poivree
        plush girafe => girafe peluche
        cheese =>  """,
    ]

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device,
    )
    print("All ok")
    out_tokens, out_text = model.text_completion(prompts, max_gen_len=64)
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f"{out_text[i]}")
        print("-" * 50)
