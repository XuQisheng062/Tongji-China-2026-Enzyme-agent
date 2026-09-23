import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader

from lightning_model import LightningDDGModel
from datasets.dataset import UniversalMutationDataset
from utils.pdb_utils import tied_featurize_mut
from omegaconf import OmegaConf


def run_inference(checkpoint_path: str, csv_path: str, output_dir: str, 
                 dataset_name: str = "Test", 
                 batch_size: int = 1, device: str = None):
    """
    inference function
    
    Args:
        checkpoint_path: model checkpoint path
        csv_path: data csv path
        output_dir: output directory
        dataset_name: dataset name
        batch_size: batch size
        device: device name
    """
    # device selection
    if device is None:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # configuration
    cfg = OmegaConf.create({
        "data": {"side_chains": False},
        "train": {"batch_size": batch_size, "num_workers": 2}
    })
    
    # load model
    print(f"load model: {checkpoint_path}")
    model = LightningDDGModel.load_from_checkpoint(checkpoint_path)
    model.eval().to(device)

    # Gradient checkpointing trades compute for memory and should remain off
    # during inference. Explicitly disable it when the model exposes the API.
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    # load dataset
    dataset = UniversalMutationDataset(csv_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: tied_featurize_mut(b)
    )
    
    # inference
    all_predictions = []
    
    print("start inference...")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"process batch {batch_idx}/{len(dataloader)}")
            
            if batch is None:
                continue
                
            # move data to device
            for key in ['wt_tokens', 'mut_tokens', 'wt_mask', 'mut_mask', 'ddg']:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            try:
                outputs = model(batch)
                all_predictions.append(outputs['ddg'].detach().cpu())
            except Exception as e:
                raise RuntimeError(f"batch {batch_idx} failed: {e}") from e
    
    # merge results
    if not all_predictions:
        raise RuntimeError("UniStab produced no predictions")
    predictions = torch.cat(all_predictions).numpy().reshape(-1)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("UniStab produced NaN or infinite predictions")

    os.makedirs(output_dir, exist_ok=True)
    results = {'predictions': predictions}
    np.savez(os.path.join(output_dir, f"{dataset_name}_results.npz"), **results)
    
    # save checkpoint info
    with open(os.path.join(output_dir, "checkpoint.txt"), "w") as f:
        f.write(f"{checkpoint_path}\n")
    
    print(f"results saved to: {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--data', type=str, required=True, help='data csv path')
    parser.add_argument('--output', type=str, default='./results', help='output directory')
    parser.add_argument('--name', type=str, default='Test', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device', type=str, default=None, help='device')
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        csv_path=args.data,
        output_dir=args.output,
        dataset_name=args.name,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
