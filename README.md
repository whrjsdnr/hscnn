# HSCNN / HSCNN+ (HSCNN-u / HSCNN-R / HSCNN-D) - PyTorch

Train & evaluate hyperspectral reconstruction from RGB.
- RGB: jpg/png
- HSI: .mat (legacy or MATLAB v7.3 HDF5)
- Robust loader: skip broken MAT files and log to `bad_mat.log`

## Install
```
pip install -r requirements.txt
```
Run (example)
python train.py ^
  --train_rgb "D:\geonug\hscnn\Train_RGB" ^
  --train_hsi "D:\geonug\hscnn\Train_Spectral" ^
  --valid_rgb "D:\geonug\hscnn\Valid_RGB" ^
  --valid_hsi "D:\geonug\hscnn\Valid_spectral" ^
  --workers 0 --train_loss mrae --out_bands 31
Notes

Use --mat_key if the MAT contains multiple arrays and auto-detection fails.

Normalization:

--mat_norm auto_divmax (default): if max>1.5 divide by max

--mat_norm fixed_div --mat_fixed_div 65535

--mat_norm none

Metrics at test: MRAE / RMSE / SAM / Spectral Gradient

