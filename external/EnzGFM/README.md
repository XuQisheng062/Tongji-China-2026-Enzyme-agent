# An Enzyme-Specific Protein Language Model for Catalytic Property Prediction

This repository contains the implementation for the paper:  
**"An enzyme-specific protein language model for catalytic property prediction"**

## Installation

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Install PyTorch and dependencies:
   ```bash
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```

3. Install additional requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Tasks

### 2.1 Kinetic Parameter Prediction

Run the following command:
```bash
python ./enzyme_kinetic_parameter/enzyme_kinetic_parameter.py \
  --model_save_dir <model_path> \
  --data_dir <data_path> \
  --output_dir <output_path> \
  --kinetic_type <parameter_type> \
  --tokenizer <tokenizer_path>
```

**Arguments:**
- `model_save_dir`: Directory containing the pre-trained model weights (default: `./EnzGFM-150M`)
- `data_dir`: Directory containing the kinetic parameter dataset (default: `./data/kcat/`)
- `output_dir`: Directory to store training results and predictions (default: `./results/kcat`)
- `kinetic_type`: Type of kinetic parameter to predict
- `tokenizer`: Path to the tokenizer

### 2.2 Enzyme-Reaction Prediction

#### Step 1: Extract Sequence Features
```bash
python ./enzyme_reaction/extract_embeddings.py \
  --model_location <model_path> \
  --data_dir <data_path> \
  --output_dir <output_path>
```

**Arguments:**
- `model_location`: Path to the pre-trained model weights (default: `./EnzGFM-150M`)
- `data_dir`: Directory containing the training and test data files (default: `./data/enzyme_smi_split/`)
- `output_dir`: Directory where the extracted embeddings will be saved (default: `./data/enzyme_smi_split/embedding/`)

#### Step 2: Process Reaction Matrix Features
```bash
python ./enzyme_reaction/process_mat.py
```

#### Step 3: Predict Enzyme-Reaction
```bash
python ./enzyme_reaction/retrieval_tfmr.py \
  --model_path <checkpoint_path> \
  --pro_embedding_type <emb_type> \
  --save_route <results_path>
```

**Arguments:**
- `model_path`: Path to the model checkpoint file (default: `./pth/EnzGFM-650M_final_model.pth`)
- `save_route`: Directory to save results and outputs (default: `./results`)
- `pro_embedding_type`: Type of protein embedding model (default: `EnzGFM-650M`)

### 2.3 EC Number Prediction

#### Step 1: Extract Sequence Features
```bash
python ./ec_pred/get_emb.py \
  --model_location <model_path> \
  --fasta_files <fasta_file1> <fasta_file2> \
  --output_dir <output_path>
```

**Arguments:**
- `model_location`: Path to the pre-trained model directory (required)
- `fasta_files`: List of FASTA files to process (space-separated)
- `output_dir`: Directory for extracted representations (default: `./emb`)

#### Step 2: Predict EC Numbers
```bash
python ./ec_predtest.py \
  --train_data <training_dataset> \
  --test_data <test_dataset1> <test_dataset2> \
  --data_dir <data_path> \
  --model_dir <model_save_path>
```

**Arguments:**
- `train_data`: Training dataset name (e.g., `split10`, `split70`, `split100`)
- `test_data`: Test dataset names (space-separated list)
- `data_dir`: Base data directory (default: `./train_data`)
- `model_dir`: Directory for model checkpoints (default: `./split10`)

### 2.4 Single Amino Acid Substitution Effect

Run the following command:
```bash
python ./point_mutation/predict_mutations.py \
  --model-location <model_path> \
  --sequence <base_sequence> \
  --input-dir <csv_directory> \
  --output-dir <results_directory>
```
### 2.5 EnzGFM-Agent

For more details, please refer to the readme.md in the EnzGFM-Agent




## Pretrained Weights

The pretrained model weights are available on Zenodo:  
[https://zenodo.org/records/22042585](https://zenodo.org/records/22042585)





**Arguments:**
- `model-location`: PyTorch model file or pretrained model name (default: `./EnzGFM-150M`)
- `sequence`: Base sequence to which mutations were applied (required)
- `input-dir`: Directory containing CSV files for deep mutational scan (default: `./DATA`)
- `output-dir`: Directory to save output files (default: `./results`)

## References

This implementation is based on/inspired by:
- [ReactZyme](https://github.com/WillHua127/ReactZyme)
- [CLEAN](https://github.com/tttianhao/CLEAN)
- [UniKP](https://github.com/Luo-SynBioLab/UniKP)
- [ESM](https://github.com/facebookresearch/esm)
