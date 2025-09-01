****************************************************************************************
necessary installation for usage:


### Train (transfer learning on in-vivo H&N data)
Run from `xQSM/python/training/`:
```bash
python Train_NoFreeze_TL.py \
  --input_root "/path/to/QSM_data" \
  --target_root "/path/to/QSM_consensus" \
  --pretrained_path "/path/to/xQSM/Pretrained_Checkpoints/xQSM_invivo.pth" \
  --snapshot_path "/path/to/xQSM/ckpt" \
  --ckpt_folder "run_name" \
  -bs 32 -ep 100 -lr 4e-4 -ps 48 [-se] \
  [--input_suffix "_unwrapped-SEGUE_mask-nfe_bfr-PDF_localfield.nii.gz"] \
  [--target_suffixes "_ConsensusMean.nii.gz"]
```

The arguments inside the [...] are optional.
- se determines if the xQSM network will have additional squeeze and excitation layers
- input and output suffixes allow for one to train the model with data which have different naming conventions that expected (i.e. using the consensus or the mean of the 3 different reconstruction methods available to us as the ground truth rather than just a single method as the ground truth) 


Notes:
- Use `-se` to enable squeeze-and-excitation.
- Multiple targets supported with comma-separated `--target_suffixes`.

### Train on synthetic data
From `xQSM/python/training/synthetic/`:
```bash
./train_synthetic.sh /path/to/simulated_volumes_nifti
# or
python train_synthetic.py --data_directory /path/to/simulated_volumes_nifti \
  --batch_size 32 --epochs 100 --learning_rate 4e-4 --patch_size 48
```

### Testing / Evaluation
Use the validation/evaluation script to compute metrics (PSNR, SSIM, xSIM) and save qualitative NIfTI outputs:
```bash
python eval/run_val_eval.py \
  --data_directory "/path/to/QSM_data" \
  --weights_path "/path/to/checkpoints/xQSM_*.pth" \
  --save_dir "eval/Inference_Results/<run_name>" \
  --split val \
  --encoding_depth 2 \
  --ini_chNo 64 \
  [--squeeze_exc]
```
Outputs: `<save_dir>/*_{input,target,output}.nii.gz` and `metrics.txt`.

### Data layout (expected)
`QSM_data/sub-XX/ses-YY/qsm/*_unwrapped-SEGUE_mask-nfe_bfr-PDF_localfield(.nii[.gz])` (inputs)
and matching targets (e.g., `_ConsensusMean.nii.gz` or specified suffixes) under the same structure.
