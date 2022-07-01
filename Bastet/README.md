# Project Bastet #
Project Bastet is focused on the implementation of a patch-wise CNN which can be used in the segementation
of electron microscopy (EM) images.

## Table of content

- [Project description](#project-description)
- [Installation](#installation)
    * [Prerequisites](#prerequisites)
    * [Libraries](#libraries)
- [Contact](#contact)

## Project description
The aim of this project is to build a patch-wise CNN which can segment large EM images, based on a
partially annotated mask given by the user. This is done by 
using a custom data loader to extract patches from an image to train a convolutional neural network to 
classify pixels.

## Installation
To install simply clone this repo and navigate to the Bastet directory.
Cloning can be done using the following command:  
```
git clone https://github.com/devalk96/Project-Ra.git
```

To install the required packages you can use:
``pip install -r requirements.txt``  
It is recommended that you install the required packages 
in a separate virtual environment.

### Prerequisites
* Python 
  * Version 3.8 or higher

### Libraries

All of the nececary libraries can be installed using the following command:  
``pip install -r requirements.txt``

|Name                                   |Version                |   
|---                                    |---                    |
|Pillow                                 |9.1.1                  |
|numpy                                  |1.21.5                 |
|torch                                  |1.8.0+cu111            |
|torchvision                            |0.9.0+cu111            |

## Running the script

The script can be run using the following command:
```
python3 bastet_net.py [-h] [-e EPOCHS] -i INPUT_FILE -m MASK [-s SEGMENT_FILES [SEGMENT_FILES ...]] [-sl SAVE]
```

|Option                                 |Explanation                                |Default    | 
|---                                    |---                                        |---        |
|-h                                     |Help                                       |           |
|-e                                     |Number of epochs to train                  |3          |
|-i                                     |Input image                                |           |
|-m                                     |Mask image, encoded with indexed colors    |           |
|-s                                     |List of images to segment                  |           |
|-sl                                    |Location to save trained network           |           |

## Contact

* K.A. Notebomer
  * k.a.notebomer@st.hanze.nl
  * [Skippybal](https://github.com/Skippybal)
