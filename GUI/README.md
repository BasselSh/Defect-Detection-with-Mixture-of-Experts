# GUI description
This GUI is designed to test the robustness of any model for different conditions applied on the input images. 
It visualizes the augmented images, and updates the configuration of the pipeline of test data, and generate a bash file containig the test commands to run testing on multiple configurations consequently.
The default augmentations are:
'none', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'motion_blur', 'zoom_blur', 'snow',
'fog', 'brightness', 'contrast', 'elastic_transform',
'pixelate', 'jpeg_compression', 'speckle_noise',
'spatter', 'saturate'

# Update GUI design
For modifying the design of the GUI using Qt Desinger, you can edit the file ``` gui.ui``` and save the changes in the same file. Then run the following comand to generate the python code of the GUI:

```bash
pyuic5 -x  gui.ui -o widgets.py
```

