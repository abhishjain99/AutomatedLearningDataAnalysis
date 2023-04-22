## Question 3:
### Pre-requisites: Make sure you have the following libraries installed
- pandas
- numpy

### To run the Python file:
1. Download the [real estate dataset](https://github.ncsu.edu/asfirodi/engr-ALDA-Fall2022-H16/blob/CSC522/HW4/adj_real_estate.csv) from our github repository.
2. Download the python code into your machine.
3. Run the python file: python3 h16_hw4_q3.py
4. After that, output should be displayed.

### Example

> python3 h16_hw4_q3.py

<hr>

## Question 4:
### Pre-requisites: Make sure you have the following libraries installed
- pandas
- numpy
- tensorflow
- matplotlib

### To run the Python file:
1. Download the [ann 2022 folder](https://github.ncsu.edu/asfirodi/engr-ALDA-Fall2022-H16/blob/CSC522/HW4/ann_2022/) from our github repository.
2. Download the python code into your machine.
3. Run the python file: python3 h16_hw4_q4.py
4. After that, output should be displayed.

### Example

> python3 h16_hw4_q4.py

### To download tensorflow on Mac (M1 Chip)
1. curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
2. bash Miniconda3-latest-MacOSX-x86_64.sh
3. source ~/.zshrc
4. conda create --name hw4
5. conda deactivate
6. conda activate hw4
7. pip3 install --upgrade pip
8. pip3 install tensorflow-macos
9. python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

If line 9 works properly, that means tensorflow is installed succesfully.

<hr>

## Question 5:
### Pre-requisites: Make sure you have the following libraries installed
- pandas
- numpy
- matplotlib
- sklearn

### To run the Python file:
1. Download the [svm 2022 folder](https://github.ncsu.edu/asfirodi/engr-ALDA-Fall2022-H16/blob/CSC522/HW4/svm_2022/) from our github repository.
2. Download the python code into your machine.
3. Run the python file: python3 h16_hw4_q5.py
4. After that, output should be displayed.

### Example

> python3 h16_hw4_q5.py