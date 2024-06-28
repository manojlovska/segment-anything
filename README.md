# Analysis of NPL TEM images using SAM

## Introduction
This project focuses on determining the diameter distribution of NPLs with optimized magnetic properties colloidally stabilized in a solvent or soft matrix, utilizing the Segment Anything Model. The original README file of the SAM project can be found under README_SAM.md.

## Steps to reproduce
### Step 1: Clone the repository
```shell
git clone git@github.com:manojlovska/segment-anything.git
```

### Step 2: Create virtual environment with conda and activate it
```shell
conda create -n env_name python=3.8.5
conda activate env_name
```

### Step 3: Install the sam module
```shell
cd segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Step 4: Install pytorch
Install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies. On a CUDA GPU machine, the following will do the trick:
```shell
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

Or if you don't have GPUs:
```shell
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
```

[Here](https://pytorch.org/get-started/previous-versions/) you can find the pytorch distribution that suits your hardware requirements.

### Step 5: Install other requirements
```shell
pip install -r requirements.txt
```

### Step 6: Download SAM weights inside a weights/ directory
```shell
mkdir weights
cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Step 7: Obtain images' masks
Run the SAM-zero-shot.py script to obtain the masks. An example of the following command:
```shell
python SAM-zero-shot.py --img_path "./data/Porazdelitev delcev original" --model-type "vit_b" --device "cuda" (or "cpu") --convert-to-rle --image-extension "tif"
```
* Notes: 
  * type the path to your dataset, or an image file you want segmented
  * --convert-to-rle flag must be present to save the masks in coco RLE format
  * the masks will be saved in "./outputs" by default, add --output "path/to/desired/output/directory" if you want to change this

Run 
```shell
python SAM-zero-shot.py --help
```
for additional information on how to use the command.

### Step 8: Compare generated and ground truth histograms
Run
```shell
python compare_histograms_main.py --images-path '/path/to/Porazdelitev delcev original' --save-hist
```
to compare the predicted and ground truth histograms for the whole dataset or
```shell
python compare_histograms_main.py --images-path '/path/to/Porazdelitev delcev original/BSHF-DBSA-210325' --save-hist
```
for a particular experimental analysis.

* Notes: 
  * Run 
  ```shell  
  python compare_histograms_main.py --help 
  ``` 
  for additional information on how to use this command.
  * If you wish you can choose different size constraints --t_min and --t_max, default are 10 nm and 150 nm
  * Add --filter-edge to filter out the edge particles (Particularly slow on non-GPU machine!)
  * If you chose different path to save the output masks of the SAM model, you have to specify that path here by using --masks-path

Generated histograms and predicted diameters in nm are saved in "./results" directory, you can change this by specifying different --save-path.

### Step 9: Generate histograms out of predicted diameters
If you don't have ground truth diameters distributions, you can plot only the predicted diameter distributions. For example, to generate diameters distribution out of one image, run the following command:
```shell
python generate_histograms_main.py --images-path '/path/to/Porazdelitev delcev original/BSHF-DBSA-210325/BSHF-DBSA-210325_0001.tif' --masks-path "/home/anastasija/Documents/IJS/E8/Magnetic-Particles/code/segment-anything/output/BSHF-DBSA-210325/BSHF-DBSA-210325_0001.json" --save-hist --ratio 0.2410
```

Or if you want to plot the diameter distribution of an experimental analysis consisted of multiple images:
```shell
python generate_histograms_main.py --images-path '/path/to/Porazdelitev delcev original/BSHF-DBSA-210325' --masks-path "/home/anastasija/Documents/IJS/E8/Magnetic-Particles/code/segment-anything/output/BSHF-DBSA-210325" --save-hist --ratio 0.2410
```
Generated histograms and predicted diameters in nm are saved in "./results" directory, you can change this by specifying different --save-path.

Here you must specify the masks path and the ratio for converting the diameters in nm.

### Step 10: Visualize the masks
If you want to check if the obtained masks for an image are well generated, run the command:
```shell
python visualize_masks.py --image-path '/path/to/Porazdelitev delcev original/BSHF-DBSA-210325/BSHF-DBSA-210325_0001.tif' --save-plot --ratio 0.2410
```

Change the size constraints by specifying --tmin and --tmax. Specify different --save-path to save the plot in different directory.

For all the scripts you can run the following command for additional information:
```shell
python script-name.py --help
```