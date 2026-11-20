## Module

::: stackgrad.module.Module
::: stackgrad.module.Sequential

## Neural Network classes

::: stackgrad.layers.Linear
::: stackgrad.conv.Conv2d
::: stackgrad.conv.MaxPool2d
::: stackgrad.conv.Flatten
::: stackgrad.layers.Embedding
::: stackgrad.layers.LayerNorm
::: stackgrad.layers.RMSNorm
::: stackgrad.layers.GroupNorm
::: stackgrad.layers.BatchNorm1d
::: stackgrad.layers.Dropout
::: stackgrad.layers.SwiGLU
::: stackgrad.attention.Attention
::: stackgrad.attention.KVCache
::: stackgrad.gpt.TransformerLM

## Activations

::: stackgrad.layers.ReLU
::: stackgrad.layers.Sigmoid
::: stackgrad.layers.Tanh
::: stackgrad.layers.GELU
::: stackgrad.layers.Softmax

## Functional

::: stackgrad.functional
    options:
        show_root_heading: false
        show_source: false

## Loss Functions

::: stackgrad.loss.mse_loss
::: stackgrad.loss.bce_loss
::: stackgrad.loss.cross_entropy_loss

## Optimizers

::: stackgrad.optim.SGD
::: stackgrad.optim.Adam
::: stackgrad.optim.AdamW
::: stackgrad.optim.cosine_lr
::: stackgrad.optim.clip_grad_norm
::: stackgrad.optim.LossScaler

## Load/Save

::: stackgrad.state.load_safetensors
::: stackgrad.state.iter_safetensors

`Module.save`, `Module.load`, `Module.state_dict`, and `Module.load_state_dict` are listed under [Module](#stackgrad.module.Module).

## Data

::: stackgrad.data.DataLoader

## Gradient checking

::: stackgrad.grad_check.numerical_gradient
::: stackgrad.grad_check.check_gradient
