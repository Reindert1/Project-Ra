# Image filter tools #
This package contains various scripts used in the pipeline for image filtering.

## Tools

- [img_to_label.py](#image_to_label)
    * [About](#About)
    * [Guide](#Packages)
  
- [image_overlayer.py](#image_overlayer.py)
    * [About](#Prerequisites)
    * [Guide](#Packages)


- [Prerequisites](###Prerequisites)
- [Packages](###Packages)
- [Contact](#contact)

## img_to_label.py  
### About 
This tool converts images to labeled data. 
Non-coloured data is labeled as 0 and coloured data is labeled as 1.

### Guide
Expects commandline input. Inputfile-path (-i) and outputfile-path (-o).  
Example: ```python3 img_to_label.py -i data/inputfile -o data/outputfile```

## image_overlayer.py
### About 
This tool overlays multiple images to a background.

Example of an overlay:   
![Example](https://github.com/devalk96/Project-Ra/blob/main/scripts/image_filters/docs/images/example.tif)



### Guide
#### Arguments 
* ```--labelmode``` (toggle) *DEPRECATED*
  * ~~This argument is used when image are being used that only contain 0/1 data. 
   The convert will automatically assign a color.~~

* ```--resize 'width' 'height'``` (not-required)
  * Resizes the output image to the provided dimensions

* ```--output 'path'``` (required)
  * Set output name. 

* ```--background 'path'``` (required)
  * Path to the image that will be used as background.

* ```--overlay 'path' 'path` 'etc..``` (required)
  * Path(s) to the images which are used as masks.

#### Basic usage: 
Images are used which are already colored. Example black and red image.
```python3 image_overlayer.py -b data/background.tif --overlay data/overlay1.tif data/overlay2.tif -O data/output.tif```


## Prerequisites
* Python 
  * Version 3.6

## Packages
| Name   | Version |   
|--------|---------|
| Pillow | 8.4.0   |
| numpy  | 1.20.3  |


## Contact
* S.J. Bouwman (**maintainer**)
  * s.j.bouwman@st.hanze.nl 

* R.F. Visser
  * r.f.visser@st.hanze.nl 

* K.A. Notebomer
  * k.a.notebomer@st.hanze.nl
