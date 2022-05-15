# Noise2Fast

# Installation
If you don't already have anaconda, install it by following instructions at this link: https://docs.anaconda.com/anaconda/install/.

It would also be helpful to have ImageJ installed: https://imagej.nih.gov/ij/download.html.

Open Anaconda Prompt (or terminal if on Mac/Linux) and enter the following commands to create a new conda environment and install the required packages:

```python
conda create --name N2F
conda activate N2F
conda install -c pytorch pytorch=1.8.0
conda install -c conda-forge tifffile=2019.7.26.2
```
If the installs don't work, removing the specific version may fix this.
# Using Noise2Fast on your 2D grayscale data

Create a folder in the master directory (the directory that contains N2F.py) and put your noisy images into it. Then open anaconda prompt/terminal and run the following:

```python
cd <masterdirectoryname>
conda activate N2F
python N2F.py <noisyfolder>
```
Replacing <masterdirectoryname> with the full path to the directory that contains N2F.py, and replacing <noisyfolder> with the name of the folder containing images you want denoised. Results will be saved to the directory '<noisyolder>_N2F'. Issues may arise if using an image format that is not supported by the tifffile python package, to fix these issues you can open your images in ImageJ and re-save them as .tif (even if they were already .tif, this will convert them to ImageJ .tif).
  
# Using Noise2Fast on your colour images, stacks and hyperstacks

To run on anything other than 2D grayscale images, use N2F_4D.py. This supports an arbitrary number of dimensions, as long as the last two dimensions are x and y. For example, here we use it on a 16x6x2x250x250 (tzcxy) image:
  
```python
cd <masterdirectoryname>
conda activate N2F
python N2F_4D.py livecells
```  

Output is in ImageJ format.

# Using Noise2Fast on provided datasets

To run N2F on the noisy confocal images, open a terminal in the master directory and run:

```python
cd <masterdirectoryname>
python N2F.py Confocal_gaussianpoisson
```
The denoised results will be in the directory 'Confocal_gaussianpoisson_N2F'.

To run N2F on our other datasets we first need to add synthetic gasussian noise. For example to test N2F on Set12 with sigma=25 gaussian noise, we would first: 
```python
cd <masterdirectoryname>
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
cd <masterdirectoryname>
python compute_psnr_ssim.py Set12_gaussian25_N2F Set12
```

You can replace 'Set12' and 'Set12_gaussian25' with any pair of denoised/ground truth folders (order doesn't matter). Average PSNR and SSIM will be returned for the entire set.
  

  
# Running compared methods

We can run DIP, Noise2Self and Ne2Ne in the N2F environment:

```python
conda activate N2F
python DIP.py Confocal_gaussianpoisson
python noise2self.py Confocal_gaussianpoisson
python Ne2Ne.py Confocal_gaussianpoisson
```

