# Attention Gym

Attention Gym is a collection of helpful tools and examples for working with [flex-attention](https://pytorch.org/docs/main/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)

[**🎯 Features**](#-features) |
[**🚀 Getting Started**](#-getting-started) |
[**💻 Usage**](#-usage) |
[**🛠️ Dev**](#️-dev) |
[**🤝 Contributing**](#-contributing) |
[**⚖️ License**](#️-license)
## 📖 Overview

This repository aims to provide a playground for experimenting with various attention mechanisms using the FlexAttention API. It includes implementations of different attention variants, performance comparisons, and utility functions to help researchers and developers explore and optimize attention mechanisms in their models.

![favorite](https://github.com/user-attachments/assets/3747fd24-1282-4d65-9072-882e55dad0ad)

## 🎯 Features

- Implementations of various attention mechanisms using FlexAttention
- Utility functions for creating and combining attention masks
- Examples of how to use FlexAttention in real-world scenarios

## 🚀 Getting Started

### Prerequisites

- PyTorch (version 2.5 or higher)

### Installation

```bash
git clone https://github.com/meta-pytorch/attention-gym.git
cd attention-gym
pip install .
```

## 💻 Usage

There are two main ways to use Attention Gym:

1. **Run Example Scripts**: Many files in the project can be executed directly to demonstrate their functionality:
   ```bash
   python attn_gym/masks/document_mask.py
   ```
   These scripts often generate visualizations to help you understand the attention mechanisms.

2. **Import in Your Projects**: You can use Attention Gym components in your own work by importing them:
   ```python
   from torch.nn.attention.flex_attention import flex_attention, create_block_mask
   from attn_gym.masks import generate_sliding_window

   # Use the imported function in your code
   sliding_window_mask_mod = generate_sliding_window(window_size=1024)
   block_mask = create_block_mask(sliding_window_mask_mod, 1, 1, S, S, device=device)
   out = flex_attention(query, key, value, block_mask=block_mask)
   ```

For comprehensive examples of using FlexAttention in real-world scenarios, explore the `examples/` directory. These end-to-end implementations showcase how to integrate various attention mechanisms into your models.

### Note

Attention Gym is under active development, and we do not currently offer any backward compatibility guarantees. APIs and functionalities may change between versions. We recommend pinning to a specific version in your projects and carefully reviewing changes when upgrading.

## 📁 Structure

Attention Gym is organized for easy exploration of attention mechanisms:

### 🔍 Key Locations

- `attn_gym.masks`: Examples creating `BlockMasks`
- `attn_gym.mods`: Examples creating `score_mods`
- `attn_gym.paged_attention`: Examples using `PagedAttention`
- `examples/`: Detailed implementations using FlexAttention

## 🛠️ Dev

Install dev requirements
```bash
pip install -e ".[dev]"
```

Install pre-commit hooks
```bash
pre-commit install
```

## 🤝 Contributing
We welcome contributions to Attention Gym, especially new Masks or score mods! Here's how you can contribute:

### Contributing Mods

1. Create a new file in the [attn_gym/masks/](attn_gym/masks) for mask_mods or [attn_gym/mods/](attn_gym/mods) for score_mods.
2. Implement your function, and add a simple main function that showcases your new function.
3. Update the `attn_gym/*/__init__.py` file to include your new function.
5. [Optinally] Add an end to end example using your new func in the [examples/](examples/) directory.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## ⚖️ License

attention-gym is released under the BSD 3-Clause License.
