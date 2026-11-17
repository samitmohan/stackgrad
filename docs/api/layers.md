# Layers

Every layer subclasses `Module`. Activations are also available as plain
functions under `stackgrad.functional` (aliased as `F`).

## Linear

::: stackgrad.layers.Linear

## Activations

::: stackgrad.layers.ReLU

::: stackgrad.layers.GELU

::: stackgrad.layers.Sigmoid

::: stackgrad.layers.Tanh

::: stackgrad.layers.Softmax

## Normalization and regularization

::: stackgrad.layers.BatchNorm1d

::: stackgrad.layers.LayerNorm

::: stackgrad.layers.Dropout

## Embedding

::: stackgrad.layers.Embedding

## Convolution

::: stackgrad.conv.Conv2d

::: stackgrad.conv.MaxPool2d

::: stackgrad.conv.Flatten

## Functional

::: stackgrad.functional
    options:
      members: [relu, gelu, sigmoid, tanh, softmax, log_softmax]
