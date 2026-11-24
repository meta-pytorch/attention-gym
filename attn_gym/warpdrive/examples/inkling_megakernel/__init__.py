"""Inkling 2 Mini (Thinking Machines' looped MoE) decode as warpdrive megakernel launches, on
the shared megakernel package (``examples.megakernel``). Modules:

- ``ref``: the PyTorch oracle (HF's eager ``Inkling2LoopForCausalLM`` numerics, except for the
  experts' declared activation quantization, which it follows by default) and checkpoint
  loading; ``act_quant``: that quantization (vLLM's SM100 quantizers) and its declaration;
- ``fast``: a CUDA-graph capturable decode step on swappable kernels;
- ``visit_kernel``: a visit (its attention half ``attn_half``, then its MoE MLP half
  ``mlp_half``) as one warpdrive launch, with message gates in core passes after the first;
  ``msg_kernel``: a pass's messages; ``head_kernel``: the final block residual, norm and LM
  head; ``common``: their shared device code;
- ``quant``: the checkpoint's NVFP4/MXFP8 experts kept quantized in the layout of the package's
  expert GEMV decode (``examples.megakernel.formats``, ``examples.megakernel.quant_gemv``), their
  exact bf16 block values and dequantization, and the oracle on such a model;
- ``megakernel``: the host side: the whole decode step as launches (``InklingDecoder``);
- ``bench``: CUDA-graph timings of each launch kind and of the decode step;
- ``chat``: the model in the megakernel chat (``attn_gym.warpdrive.examples.megakernel.chat``).
"""
