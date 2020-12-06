# PianoInterpreter
This projects simulate a piano using just your hand,python3 ,opencv and haar classificator.
# Environment
I have tested on Ubuntu 18.04. The code may certainely  work on other systems.
* Ubuntu 18.04
* Python3.6
* Opencv3
* Numpy
# Instalation
Follow the steps:

```
git clone https://github.com/habbichelotfi/piano_interpreter

cd piano_interpreter

python3.6 main.py
```
# Explaining the process:
### first step:Detect the piano
For that i used haar classificator and i tranined the data using haar cascade trainer.

For haar cascade trainer you need to create a repostory that inculde positive images(the target) and negatives images and let the magic process

link to:https://amin-ahmadi.com/cascade-trainer-gui/

after the creation of the trained classifor we import it 
