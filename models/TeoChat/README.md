---
title: SatQuery AI TEOChat Specialist
sdk: gradio
app_file: app.py
---

# TEOChat Specialist

TEOChat is a downstream specialist called by Qwen for multi-image temporal
or cross-modal inference. The existing VideoLLaVA/TEOChat model, processor,
conversation handling, and interleaved multi-image inference remain intact.

## Service contract

The Gradio API is `/teochat` and accepts four optional ordered image inputs,
one specialist prompt, and a system-controlled generation limit:

```text
image_1, image_2, image_3, image_4: file inputs
prompt: specialist instruction
max_new_tokens: hidden system configuration, 1..1024
```

At least one image is required. Multiple images are passed together to one
`TEOChatEngine.analyze` call, preserving the model's interleaved temporal
behavior. TEOChat does not route workflows, access Firebase, create signed
URLs, or synthesize the final user answer.

The endpoint contract is explicit:

```text
image_t1 = optical image at T1
image_t2 = optical image at T2
```

Requests with either image missing are rejected intentionally.

## ZeroGPU model loading

The Hugging Face ZeroGPU deployment loads TEOChat with FP16 weights and no
bitsandbytes quantization:

```text
TEOCHAT_QUANTIZATION=none
load_8bit=False
load_4bit=False
torch_dtype=torch.float16
```

For local deployments with sufficient memory, `TEOCHAT_QUANTIZATION=8bit` or
`4bit` preserves the existing opt-in quantized loader. The compatibility alias
`TE0CHAT_QUANTIZATION` is also accepted. The FP16 path does not require
bitsandbytes.

## Structured result

The endpoint returns JSON with `answer`, `bounding_boxes`, `input_regions`,
and `evidence`. Legacy model output such as `[20, 30, 70, 80]` becomes an
image-associated canonical box:

```json
{
  "image_index": 0,
  "box": {
    "x_left": 20,
    "y_top": 30,
    "x_right": 70,
    "y_bottom": 80,
    "angle": 0
  }
}
```

Canonical `{x_left,y_top,x_right,y_bottom|angle}` output is also preserved.
Axis-aligned legacy boxes use angle `0`; no confidence or spatial evidence
is fabricated. Multiple boxes and image/frame markers are preserved.

The existing FastAPI-style model service is not required by the Qwen adapter;
`app.py` is the specialist Gradio entrypoint. Signed URLs are converted to
temporary files by the Gradio client transport, outside TEOChat authorization.


<p align="center">
    <img src="static/logo.png" width="150" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="http://arxiv.org/abs/2410.06234">TEOChat: Large Language and Vision Assistant<br> for Temporal Earth Observation Data [ICLR 2025]</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub and watch 👀 for the latest updates.  </h2>


<h5 align="center">

[![hf_dataset](https://img.shields.io/badge/🤗-Dataset-blue.svg)](https://huggingface.co/datasets/jirvin16/TEOChatlas)
[![hf_model](https://img.shields.io/badge/🤗-Model-blue.svg)](https://huggingface.co/jirvin16/TEOChat)
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/jirvin16/TEOChat)  <br>
[![arXiv](https://img.shields.io/badge/Arxiv-2410.06234-b31b1b.svg?logo=arXiv)](http://arxiv.org/abs/2410.06234)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/ermongroup/TEOChat/blob/main/LICENSE)

</h5>

## 📰 News
* **[2025.3.25]** 📊 Validation code for the detection datasets and code for fine-tuning TEOChat is now available! Please see [TRAIN_AND_VALIDATE.md](TRAIN_AND_VALIDATE.md) for validation and fine-tuning instructions.
* **[2025.3.10]** 🔍 Inference and validation code (for classification datasets) is now available! Please see the [Inference](#-inference) section for inference code and [TRAIN_AND_VALIDATE.md](TRAIN_AND_VALIDATE.md) for validation instructions.
* **[2025.1.22]** 🔥 Our work has been accepted to **ICLR 2025**!
* **[2024.11.11]** 🎉 The [TEOChatlas Dataset](https://huggingface.co/datasets/jirvin16/TEOChatlas) is now available, along with model training code with instructions described in [TRAIN_AND_VALIDATE.md](TRAIN_AND_VALIDATE.md)! Validation code and instructions will be provided soon.
* **[2024.10.9]** 👀 [Paper](https://arxiv.org/abs/2410.06234), [TEOChat Model](https://huggingface.co/jirvin16/TEOChat), and 🤗 [Hugging Face demo](https://huggingface.co/spaces/jirvin16/TEOChat) are available! Please feel free to **watch** 👀 this repository for the latest updates.

## Table of Contents

* [Highlights](#-highlights)
* [Demo](#-demo)
* [Main Results](#-main-results)
* [Requirements and Installation](#%EF%B8%8F-requirements-and-installation)
* [Training & Validating](#%EF%B8%8F-training--validating)
* [Inference](#-inference)
* [License](#-license)
* [Acknowledgement](#-acknowledgement)
* [Citation](#%EF%B8%8F-citation)

## 😮 Highlights 

**TEOChat** is the first language and vision assistant that can engage in conversation about sequences of temporal earth observation imagery, and exhibits impressive performance on multiple temporal instruction-following tasks.

<img src="assets/figure1.png"/>

### 📚 TEOChatlas: A new instruction-following dataset for temporal EO data
We introduce a new instruction-following dataset for temporal EO data called **TEOChatlas** which we use to train TEOChat. TEOChatlas contains 554,071 examples spanning dozens of temporal instruction-following tasks.

<img src="assets/figure2.png"/>

### 🤖 TEOChat: A new vision-language model for temporal EO data
We design TEOChat to use a LLaVA-style architecture, combining a temporally shared vision encoder with a LLaMA 2 LLM connected through an MLP vision-language projector

<img src="assets/figure3.png"/>

## 🤗 Demo

### Gradio Web UI

We provide an [online demo](https://huggingface.co/spaces/jirvin16/TEOChat) in Huggingface Spaces.

You can also run the demo locally by running the following command:
```bash
python videollava/serve/teochat_demo.py 
```

<img src="assets/demo.gif" width="500" />

## 🚀 Main Results
We demonstrate that TEOChat:
- outperforms a previous VLM for single EO images ([GeoChat](https://github.com/mbzuai-oryx/geochat)) and a VLM for temporal sequences of natural images ([Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)), and also rivals specialist models on multiple tasks.
- achieves impressive zero-shot performance on an EO change detection and a change QA dataset.
- outperforms two strong proprietary foundation models for modeling sequences of images (GPT-4o and Gemini-1.5 Pro).
- possesses strong single image capabilities, outperforming GeoChat on multiple zero-shot scene classification and visual question answering tasks.

### Temporal Tasks
<p align="left">
<img src="assets/table1.png" width=80%>
</p>

### Zero-shot Temporal Tasks and Comparison with Proprietary Foundation Models
<p align="left">
<img src="assets/table3-4.png" width=80%>
</p>

### Single Image Tasks
<p align="left">
<img src="assets/table5.png" width=80%>
</p>


## 🛠️ Requirements and Installation
* Python >= 3.9
* Pytorch == 2.2.1
* CUDA Version >= 12.1
* Install required packages:
```bash
git clone https://github.com/ermongroup/TEOChat.git
cd TEOChat
conda create -n teochat python=3.9 -y
conda activate teochat
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install git+https://github.com/facebookresearch/pytorchvideo
```

## 🗝️ Training & Validating
The training & validating instructions, including how to download the TEOChatlas dataset, are in [TRAIN_AND_VALIDATE.md](TRAIN_AND_VALIDATE.md).

## 🧠 Inference
You can use the following code to run inference with TEOChat on GPU:
```python
from videollava.eval.eval import load_model
from videollava.eval.inference import run_inference_single

tokenizer, model, processor = load_model(model_path="jirvin16/TEOChat", model_base=None, load_8bit=True, device='cuda')

# A list of image paths, all are fed into the model.
image_paths = ["videollava/serve/examples/xBD_cls_1.png", "videollava/serve/examples/xBD_cls_2.png"]
# Note you must include the video tag <video> in the input string otherwise the model will not process the images.
inp = "These are two satellite images in chronological order: <video> Classify the level of damage experienced by the building at location [0, 8, 49, 53]."

response = run_inference_single(model, processor, tokenizer, inp, image_paths)
print(response)
```

## 👍 Acknowledgement
* [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) The codebase and model we built upon.
* [GeoChat](https://github.com/mbzuai-oryx/geochat) The single image instruction-following dataset we included in TEOChatlas.

## 🔒 License
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/ermongroup/TEOChat/blob/main/LICENSE) file.
* The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@inproceedings{irvin2024teochat,
  title={TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data},
  author={Irvin, Jeremy Andrew and Liu, Emily Ruoyu and Chen, Joyce Chuyi and Dormoy, Ines and Kim, Jinyoung and Khanna, Samar and Zheng, Zhuo and Ermon, Stefano},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
