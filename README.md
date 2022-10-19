[![.NET](https://github.com/horczech/BrnoAiDaysWorkshop/actions/workflows/dotnet.yml/badge.svg)](https://github.com/horczech/BrnoAiDaysWorkshop/actions/workflows/dotnet.yml)

# BrnoAiDaysWorkshop


## Schedule
 1. Intro presentation (10 min)
 2. Setup of environment (15 min)
 3. ModelBuilder + AutoML (15 min)
 4. Guided demonstration of ML.NET (20 min)
 5. Individual tasks (60 min)
 6. Enjoy your trained models!
     1. Try console application **BrnoAiDaysWorkshop.TryIt** 
     2. Try web API application **BrnoAiDaysWorkshop.TryIt.MiniAPI**

## Setup 
 1. Connect to wifi. 
     1. Connect to YSoft-Visitors wifi
     2. Browser with registration opens, go to Registration section
     3. Fill your name, your valid email where you will get confirmation code, and input vaclav.novotny@ysoft.com into fourth input.
    ![YSoft Visitors registration](https://github.com/vaclavnovotny/images/blob/main/ysoftwifi.png)

 2. Clone repository
     1. Open Visual Studio, click Continue without code 
     2. Go to menu item Git -> Clone
     3. Fill repository location as: https://github.com/horczech/BrnoAiDaysWorkshop.git
     4. Click clone.
 
 3. Run the BrnoAiDaysWorkshop.Intro
     1. Open solution file **BrnoAiDaysWorkshop.sln** at cloned repository location
     2. Go to menu item Build -> Build Solution
     3. Find project BrnoAiDaysWorkshop.Intro in solution explorer, right click and hit Set as startup project
     4. Push F5, debug should start

## Tasks
obecny povidani

 ### Task 0 (5 min)
 When you are training ML model, its important to shuffle the input data so the model will see random class during each step of training. Use the [ML.NET documentation](https://learn.microsoft.com/en-us/dotnet/machine-learning/), and figure out how to shuffle the loaded data in the `data` variable. 
 
 ### Task 1 (5 min)
 When you are training a model, you want save approximately 20% of the dataset for testing. This testing data wont be used for training at all so you can use it for evauation of your model since it will be first time the model will see these data. Figure out how to split the dataset loaded in the `dataset` variable so the 80% of the data is used for training and the rest for model evaluation.
 
 ### Task 2 (30 min)
 ### Task 3 (10 min)
 ### Task 4 (15 min)
 
 
## Helpful links
 - [ML.NET documentations](https://learn.microsoft.com/en-us/dotnet/machine-learning/)
 - [Available trainers](https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-choose-an-ml-net-algorithm)
 - [Available transformations](https://learn.microsoft.com/en-us/dotnet/machine-learning/resources/transforms)
 - [Sample codes](https://github.com/dotnet/machinelearning-samples)
