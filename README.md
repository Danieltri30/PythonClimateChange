<<<<<<< HEAD
# PythonClimateChange
Our Climate Change project
=======
<img src="https://i.imgur.com/AqlOpg6.png" width="300" height="">

# Python Climate Change Project

#### Python-based tool that analyzes and visualizes climate change data made for CIS4930.

# PLEASE NOTE #
# The uncleaned datasets we used for this project are extremly large , therefore please click #
# the link below to access a cloud storage containing the .csv's which are uncleaned #
# https://drive.google.com/drive/folders/19hV-AirPnow0ZuEnzPqcD6YdM63J0XjN?dmr=1&ec=wgc-drive-globalnav-goto # 
# Please note a clean version of this data has been provided, if your Central Processing Unit is not up to date, #
# then it is advised for you to simply used the cleaned data found in the data folder with the title :  #
# FinalProcessedData.csv #

## climate_analyzer_project Structure 
#### (Placeholder - Will update when we get closer to turn-in time)
* `.venv`: Virtual Enviroment
* `data/`: Climate data (CSV format)
* `models` : Holds multiple variations of the models we trained and saved 
* `src/`: Source code & Flask Files
    * `__pycache__`: ***Ignore it*** - Stores compiled bytecode
    * `static`: **Flask** - Holds CSS & Javascript files
        * `css`: 
        * `js`:
    * `templates`: **Flask** - Holds HTML Files
* `tests/`: Unit tests
*  `tuner_dense`: Holds learning trials from algorithm tuner
* `README.md`: Project documentation
* `requirements.txt`: Project dependencies
* ResearchPaper19.pdf : our research report

## Project features
- Future climate prediction
- GUI using Flask
- Operational from backend if needbe
- CLustering performance visualization
- Neural Network performance vizualization


# Installation & Dependencies

## Create Virtual Enviorment

## Note that these steps may not work on windows, will fix this issue soon. Be assured that the MAC/Linux Operating System will run the website correctly.

### Windows:
```
    python -m venv .venv
    venv\Scripts\activate
```

### Linux/MacOS:
```
python3 -m venv .venv (Note if python3 does not work attempt to use python)
source .venv/bin/activate
```
## Building Flask Application for Development:
#### Step 1: Enter the src directory.
```
MAKE SURE YOU ARE IN THE DEFAULT DIRECTORY(Root):  'PYTHONCLIMATECHANGE'
```

#### Step 2: Install Dependencies

```
pip install -r requirements.txt
```


#### Step 3 :

```
Make sure you are in the top level of the
project directory: "PythonClimateChange"

```



#### Step 4: Run this to run our interactive gui and see the features of our project
```
PYTHONPATH=src flask --app main.py run
```
### Note we did not have enough time to add a return to main menu feature for each button,
### Simply press the back arrow to return to main menu on your browser


#### In your terminal you should see something like this:
```
(.venv) PS D:\Obsidian\PythonClimateChange\src> flask --app main.py run
 * Serving Flask app 'main.py'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
```
#### Step 5: Copy and Paste the into your preferred web browser the IP address that shows up in your terminal.
```
http://127.0.0.1:5000
```
#### If changes are made to the code, you should be able to refresh the web page to see the changes.
## use ctrl+c to shut down the Flask Development Server.

---
|             |    Random User-Guide Links         |               |
|    :----:   |    :----:   |    :----:     |
|[Scikit](https://scikit-learn.org/stable/install.html)|[Seaborn](https://seaborn.pydata.org/tutorial.html)| [Matplotlib](https://matplotlib.org/stable/users/index.html)|
|[Pandas](https://pandas.pydata.org/docs/user_guide/index.html)|[NumPy](https://numpy.org/doc/stable/)|[Flask](https://flask.palletsprojects.com/en/stable/)|
|[GitHub Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)|[Markdown Cheat Sheet](https://www.markdownguide.org/basic-syntax/)| [Requests - HTTP library](https://requests.readthedocs.io/en/latest/)
>>>>>>> 8547d7057489fc0c12c3182b88ff46c6dbe835a5
