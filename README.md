# Digit Recognition Using SNN

The overall progress of the project should be an iterative exploration of different models and 
encodings, with each iteration giving more insight into how the model classifies the digits. As 
you work on this project, keep the following questions in mind. How is the input digit encoded 
(rate coding, temporal coding, whole image, sliding window, etc.). How does the behavior of 
the models change from generation to generation as I evolve them? How does the network 
structure or inclusion/exclusion of inhibitory neurons influence the model? How do different 
parameters in the neurons or the network affect the outcome of the evolution? A suggested 
workflow could be: 

### Model selection

* Integrate and fire neurons (leaky or not). Use excitatory neurons (inhibitory neurons 
are optional). Evolve the parameters of the neurons.  
* Different  types  of  connectivity  can  be  used.  Evolve  the  network  parameters.  How 
many neurons and connections do you need? If you include inhibitory neurons, what 
is the ratio between excitation and inhibition? 

### Assignment Tasks

1. Familiarize yourself with the neurons and their parameters. Understand the components and how they influence each other’s behavior.
2. Test an encoding method to input the MNIST digits.  
3. Select parameters to evolve for your chosen neurons and network and evolve the model.  
4. Evaluate the behavior of the output model. 
5. Consider alternative methods to encode, train or perform the task.

## Environment

The app can be ran locally and any IDE can be used but it is built with Visual Studio Code in mind.

### Getting started

1. Clone the repository
2. Install the dependencies with `pip install -r requirements.txt`
3. Cuda should be installed on your machine, follow he instructions [here](https://developer.nvidia.com/cuda-toolkit-archive) to install version 11.6.
4. Run the project with `PYTHONPATH=. python src/main.py`
5. You can now change code in `main.py` to run the project with different parameters.

NB: For running the code efficiently, it is recommended to use an nVidia GPU with 16GB of memory or more, and ideally a good amount of RAM (16GB).
You might need to set `in_memory=False` in `src/main.py` if you don't have enough RAM. 

## Collaborators

Oriana Presacan - s372073@oslomet.no\
Rikard Haukemyr Donnelly - s188111@oslomet.no\
Frencis Balla - s371513@oslomet.no\
Jackson Herbert Sinamenye - s371140@oslomet.no
