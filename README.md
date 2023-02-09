# Electrical signal analysis : SARS-CoV-2 infected organoids

todo:
- [ ] [visuals](#visuals-and-resulting-figures)
- [ ] [installation](#installation)
- [ ] [data acquisition](#data-acquisition)
- [ ] [data formatting](#data-formatting)
- [ ] [usage exemples](#usage)
- [ ] [licence](#licence)

## Description
Its aim is to provide signal processing and machine learning solutions for electrical signal analysis. 
In this specific case it has been used on human brain organoids. It allows the user to use different analysis and 
data processing procedures. In those you can find smoothing, Fast-Fourier-Transform, data augmentation algorithms and others.
Those procedures have been optimized for this very project in this repository, so you may want to adapt it in many ways for your own usage.

For more information on the possible usages, please refer to the [corresponding section](#usage).

You can also check out [other repositories with a similar use](#other-repositories-with-similar-use)


## Development context
This project is developed in the context of public research in biology. It has been developed as support for the 
publication <ins>_**put publication reference**_</ins>.

## Visuals and resulting figures
<img src="https://github.com/WillyLutz/sars-cov-organoids/blob/main/Figures/Fig2b%20Zoom%20in%200-500Hz.png" width=250 height=250>

## Installation
This project has been developed under Linux-Ubuntu system, and has not been tested on other systems. It may work on other systems however.

## Data acquisition
The signal has been recorded at 10000 Hz, with a MEA 60 channels electrode. 
For more information about the array, refer to [their page (add link)](#data-formatting). Each recording has been done 3 times,
on a minimum of 3 organoids per test batch.


## Data formatting 
For most (if not all) of the analysis, a certain data format will be needed.


### Project organization
```bash
├── sars-cov-organoids
│   ├── scripts
│   │   ├── complete_procedures.py
│   │   ├── data_processing.py
│   │   ├── machine_learning.py
│   │   ├── main.py
│   │   ├── PATHS.py
│   │   ├── signal_processing.py
│   ├── venv

```

### Organizing the data
To use efficiently the project, a certain architecture will be needed when organizing the data.
```bash
├── base
│   ├── DATASET
│   ├── MODELS
│   ├── RESULTS
│   │   ├── Figures Paper
│   │   │   ├──myfigures.png
│   │   ├──myfigures.png
│   ├── DATA
│   │   ├── drug condition*
│   │   │   ├── recording time**
│   │   │   │   ├── cell condition***
│   │   │   │   │   ├── samples****
│   │   │   │   │   │   ├── myfiles.csv*****
```
* E.g. '-Stachel', '+Stachel'

** Must follow the format T=[time][H/MIN]. E.g. 'T=24H', 'T=0MIN', 'T=30MIN'.

*** What you want to classify. E.g. 'INF', 'NI'. 

**** The sample number. E.g. '1', '2'... 

***** The files that contain the data. They must follow a [certain format](#data-format).

In the data folder, you can multiply every directory as much as you have conditions.

### Data format
Across all the analysis, multiple data type will be generated. For all the files generated, it is recommended to keep tracks of the
different conditions of this very data in the file name.

#### Raw file
```2022-09-16T14-02-25t=24h NI1 STACHEL_D-00145_Recording-0_(Data Acquisition (1);MEA2100-Mini; Electrode Raw Data1)_Analog.csv```

## Development specification
Language: Python 3.10
OS: Ubuntu 22.04.1 LTS

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
For any support request, you can either use this project issue tracker, or state your request at <willy.lutz@irim.cnrs.fr> 
by precising in the object the name of this repository followed by the summarized issue.

## Contributing
This project is open to any suggestion on the devlelopment and methods. 

## Authors and acknowledgment
Author: Willy LUTZ

Principal investigator: Raphaël Gaudin

Context: MDV Team, IRIM, CNRS, Montpellier 34000, France.

## License
For open source projects, say how it is licensed.

## Project status
on going

## Other repositories with similar use
From the same author:
- [Electrical signal analysis :Tahynavirus infected brain slices](https://github.com/WillyLutz/tahynavirus-electrical-analysis)
