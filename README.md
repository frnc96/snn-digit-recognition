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

The app can be ran locally and any IDE can be used but it is built with Docker and Visual Studio Code in mind.

### Getting started

1. Start Docker Desktop
2. Make sure you are in the root folder of the app
3. Run `cd .devcontainer && docker compose up` in the terminal

Now you should see a development container called devcontainer in your Docker Desktop app.

### Developing

For development the Docker VSCode plugin is needed. Make sure that the container is running (it should be green in docker desktop, if not click the play button) and navidate to it through the plugin, right click on it and select "Attach to visual studio code". A new VSCode window should open with the app files in the project tree explorer. Happy developing :D

## Collaborators

Oriana Presacan - s372073@oslomet.no\
Rikard Haukemyr Donnelly - s188111@oslomet.no\
Frencis Balla - s371513@oslomet.no\
Jackson Herbert Sinamenye - s371140@oslomet.no