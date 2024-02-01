# threshfinder

## 1. Setup Instructions
You will first have to do an installation of the required dataset, the process for this can be found here: [LINK](https://github.com/mraoaakash/threshfinder/blob/main/data/README.md). Once you have the data downloaded, you will have to place it into the data folder. The data folder should be placed in the root directory of the project. The data folder should have the following structure:
```
threshfinder
├─ data
    ├─ NuclsEvalSet
```


## 2. Data Organization
After your data is placed into your data folder, you will have to organize it into the model input structure. To do this, you may use the  ```organize_nucls.sh``` script. This script will take in the path to your data folder and the path to your output folder. The output folder will be created if it does not exist. The script will then organize your data into the model input structure. The script can be run as follows:
```
bash runfiles/organize_nucls.sh
```

## 3. Model Training
TBD