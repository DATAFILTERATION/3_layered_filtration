# 3_layered_filtration
Location Filtration + Content Filtration+ numeric Filtration of input data


Introduction
------------

This project was done with an intention to filter the situational textual data after a disaster scenario.
This program will filter data based on Location, content as well as numeric.

Requirement
```````````````````
python 3.6 or above
NLTK
pandas
numpy
pickle
matplotlib
sklearn
pydictionary

File information
````````````````````````````
"input.csv" :                               is the input file received after parsing

"dcw.txt"                                   is the disaster content word list

"content_input.txt"                         is the input to content filtration

"Final_output.txt"                          is the output from content filtration module and input to report generation

"sorted_clustered.csv"                      is the file after location filtering which contains clusters in sorted manner.

"num_extract.csv"                           is the file generated after numeric extraction which contains the disaster related word along with their 				            number of occurances.

"exponential_output.csv"                    contains the disaster related word and their smoothed values, there may be multiple occurances.

"final_filtered_exponential_outputp.csv"    contains single disaster related word along with their number of occurances.

Thanks:-
Arindam Dutta
Rohan Basu Chowdhury
