# W-O-Name

Repository containing templates for computer vision tasks.

The Python version to be used for all tasks is **3.11**.

The folder structure is organized as follows:

* **Core**: Main framework functionalities, including:

  * **Engine**: Training and inference processes.
  * **Graph**: Model representation and computational graphs.
  * **Plugins**: External modules and additional utilities.
  * **Registry**: Model registration and management.

* **Vision**: Proposed computer vision tasks.

* **Architectures**: Backbone, Neck, Decoders, and Heads.

* **Losses**: Loss functions.

* **Evaluators**: Evaluation metrics.

* **Layers**: Primitive layers, e.g., `DoubleConv2D`, `ConvTranspose`, `DilatedConv`, etc.

The supported configuration formats for each task are:

* YAML
* TOML
* JSON
* Python
* CLI
* Web UI

## Goals

* Implementation of **lazy registration**.
