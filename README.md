# Project Ra #
Project Ra is focused on trying to build an application that can annotate electron microscopy images based on user input.

## Table of content

- [Project description](#project-description)
- [Directory descriptions](#directory-descriptions)
- [Installation](#installation)
    * [Prerequisites](#prerequisites)
    * [Packages](#packages)
    * [Running the notebooks](#running-the-notebooks)
- [Contact](#contact)

## Project description
This aim of this project is to build an application that can annotate electron microscopy data based on the user input. The goal is to make the application in such a way that a researcher can simply annotate a few cell components, and the application will learn which parts of the cell to annotate. After learning what to annotate, the application should annotate the entire image, and display the results back to the user.

Machine learning algortihms will play a vital role in the successful completion of this project.

## Directory descriptions

|Name                                       |Contains                               |   
|---                                        |---                                    |
|Bastet                                     |Patch-wise CNN implementation          |
|EyeOfRa                                    |GUI                                    |
|Ipy                                        |Image segmentation pipeline            |
|notebooks                                  |Notebooks used for research            |
|presentations                              |End of sprint presentations            |
|scripts                                    |Scripts used for research              |
|testing environments                       |Test code for research                 |

## Installation
To install simply clone this repo using the following command:  
``git clone https://github.com/devalk96/Project-Ra.git``

### Prerequisites
* Python 
  * Version 3.8 or higher
* Jupyter Notebook

### Packages
|Name                                   |Version              |   
|---                                    |---                  |
|Pillow                                 |8.4.0                |
|opencv-python                          |4.5.5.62             |
|numpy                                  |1.20.3               |
|sklearn                                |1.0.2                |
|joblib                                 |1.1.0                |
|matplotlib                             |3.5.0                |

### Running the notebooks

To run the notebooks you can either open them in Visual Studio Code, or you can start jupyter notebook by typing ``jupyter notebook`` in your terminal and navigate to the notebooks from there. 

**Important**  
The data used in the notebooks is currenty located on the assemblix server of the BIN faculty. To get access to this data please contact the repo owners.

## Contact

* S.J. Bouwman
  * s.j.bouwman@st.hanze.nl 

* R.F. Visser
  * r.f.visser@st.hanze.nl 

* K.A. Notebomer
  * k.a.notebomer@st.hanze.nl
