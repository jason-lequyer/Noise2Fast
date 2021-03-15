# Noise2Fast

# Installation
If you don't already have anaconda, install it by following instructions at this link: https://docs.anaconda.com/anaconda/install/.

# Installing Noise2Fast

Create a new conda environment and install the required packages:

```python
conda create --name N2F
conda activate N2F
conda install -c pytorch pytorch=1.8.0
conda install -c conda-forge tifffile=2019.7.26.2
```
If the installs don't work, removing the specific version may fix this.
# Using Noise2Fast on your data

Create a folder in the master directory and put your noisy images into it. Then open a terminal in the master directory and run the following:

```python
conda activate N2F
python N2F.py XXXXX
```
Where 'XXXXX' is the name of the folder containing images you want denoised. Results will be saved to the directory 'XXXXX_N2F'. Issues may arise if using an image format that is not supported by the tifffile python package, to fix these issues you can open your images in ImageJ and re-save them as .tif.

# Using Noise2Fast on provided datasets

To run N2F on the noisy confocal images, open a terminal in the master directory and run:

```python
python N2F.py Confocal_gaussianpoisson
```
The denoised results will be in the directory 'Confocal_gaussianpoisson_N2F'.

To run N2F on our other datasets we first need to add synthetic gasussian noise. For example to test N2F on Set12 with sigma=25 gaussian noise, we would first: 
```python
python add_gaussian_noise.py Set12 25
```
This will create the folder 'Set12_gaussian25' which we can now denoise:

```python
python N2F.py Set12_gaussian25
```
Which returns the denoised results in a folder named 'Set12_gaussian25_N2F'.

# Calculate accuracy of Noise2Fast

To find the PSNR and SSIM between a folder containing denoised results and the corresponding folder containing known ground truths (e.g. Set12_gaussian25_N2F and Set12 if you followed above), we need to install one more conda package:

```python
conda activate N2F
conda install -c anaconda scikit-image=0.17.2
```

Now we measure accuracy with the code:
```terminal
python compute_psnr_ssim.py Set12_gaussian25_N2F Set12
```

You can replace 'Set12' and 'Set12_gaussian25 Set12' with any pair of denoised/ground truth folders (order doesn't matter). Average PSNR and SSIM will be returned for the entire set.

# Running compared methods

We can run DIP and Noise2Self in the N2F environment:

```python
conda activate N2F
python DIP.py Confocal_gaussianpoisson
python noise2self.py Confocal_gaussianpoisson
```

However Self2Self is tensorflow based and requires us to install more conda packages to work:

```python
conda activate N2F
conda install -c conda-forge tensorflow
conda install -c conda-forge opencv=4.5.1
conda install -c anaconda keras=2.3.1
```

