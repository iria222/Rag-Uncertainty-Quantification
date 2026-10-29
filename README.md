# Rag-Uncertainty-Quantification

## Requirements
- Python 3.12 or a compatible version.
- Python dependencies listed in `requirements.txt`.
- Some dependencies, particularly those related to GPU acceleration, may require additional platform-specific configuration.

Running the project on a device equipped with a GPU is highly recommended, as the language models (LLMs) run locally.

The required hardware resources depend on the selected models, quantization settings, and available memory. Although CPU execution may be possible for the smaller models, it will be considerably slower.

## Setup

Clone the repository:
```bash
git clone https://github.com/iria222/Rag-Uncertainty-Quantification.git
cd rag-uncertainty-quantification
```

Create and activate a virtual environment using your preferred envornment manager.

Example using conda:
```bash
conda create -n rag-tfm python=3.12
conda activate rag-tfm
```

Install the dependencies:
```bash
python -m pip install -r requirements.txt
```

Install the project in editable mode:
```bash
python -m pip install -e .
```

The editable installation allows the package to be imported from the development environment without manually adding the source directory to PYTHONPATH.

## Data

This repository does not include any data files. To use the RAG pipeline, you must first add the PDF documents that will serve as the knowledge source to the data/ directory.

The system processes these documents to build the corpus and retrieved relevant information.

## Usage

### Configuration

Before running the pipeline, you can customize its behaviour by editing config.py. This module contains the main configuration settings, including:
- **Pipeline settings**: enable or disable self-evaluation, debug mode and on-demand model loading.
- **Chunking**: configure the chunk size, overlap, and cache version.
- **Models**: select the embedding, reranking, and generation models, as well as the quantization level.
- **Execution**: configure the computation device, checkpoint save frequency, and worker count.
**Directories**: specify the locations used for input data, generated outputs, and cached files.

The default device is selected automatically based on CUDA availability. Other settings, such as model names and quantization, can be adjusted to match the requirements and hardware available.

### Running the pipeline

The pipeline supports two execution modes: batch mode and interactive mode.

#### Batch mode
The pipeline processes multiple questions from a JSON file and saves the results to disk. It also supports checkpoints, allowing previously processed questions to be skipped when resuming an interrupted run.

Run the pipeline with an input JSON file:

```bash
python -m rag_uncertainty --input_json {questions.json}
```

You can specify the output filename using --ouput:
```bash
python -m rag_uncertainty --input_json {questions.json} --output {results.json}
```

Results are saved in `output/` by default, in a subdirectory named after the input file. Retrieval checkpoints are also stored there to support resuming interrunpted runs.

#### Interactive mode

In interactive mode, you can ask questions directly through the terminal, one at a time. Run the pipeline without specifying --input_json:

```bash
python -m rag_uncertainty
```

Enter a question when prompted. The system will generate an answer and display the most relevant retrieved text fragments.

To exit interactive mode, type /salir or /exit, or use Ctrl+C.



## Disclaimer
This project is intended for research and evaluation purposes only. It has not been validated as clinical decision-support software. Its outputs must not be used as medical advice or as a substitute for professional clinical judgement.