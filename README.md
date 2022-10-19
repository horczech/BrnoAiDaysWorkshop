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
Your task is to take advantage of your newly gained ML.NET skills and put it into practice by fixing a ML pipeline that can classify checkbox state (on/off). In the cloned repository is `BrnoAiDaysWorkshop` project with `Program.cs` where you can find the whole pipeline. The problem is that some of the lines in the pipeline are missing. The missing lines are marked as `todo TASK 0`. Your job is to solve all 4 tasks code and **MAKE THIS CODE GREAT AGAIN!**     

 ### Task 0 (5 min)
 When you are training ML model, its important to shuffle the input data so the model will train on random order of classes. Use the [ML.NET documentation](https://learn.microsoft.com/en-us/dotnet/machine-learning/), and figure out how to shuffle the loaded data in the `data` variable. 
 
 ### Task 1 (5 min)
 When you are training a model, you want save approximately 20% of the dataset for testing. This testing data wont be used for training so you can use it for model evaluation because it will be first time the model will see these data. Your task is to split the dataset loaded in the `dataset` variable so 80% of the data is used for training and the rest for model evaluation.
 
 ### Task 2 (30 min)
 Preprocessing and training pipeline is the center point of the ML.NET universe. When you select a ML model that you want to use for your task, you have to specify several training parameters and transform the input data to a format that is expected by the ML model that is used. In this task you will have to convert the label name (on/off) into a `key` type, connect preprocessing and training pipeline, configure the trainer and save the trained model.
  
 ### Task 3 (10 min)
 After you train your model you only know how does the model works with the training data but you have no clue if it will work with data that were not used durint the tarining. In some cases, especially when you use huge networks with few data, the model can memorize each image instead of learning general features. In those cases the accuracy on training data is very good, but the model fails when it gets new data. In this task you have to load your model, do the predictions on `test` dataset (that was not used for training) and evaluate how well model works with new data.
 
 ### Task 4 (15 min)
 Machine learning is very data hungry and you ralely have enough of data. Luckily you can use image augmentation and modify the images in your dataset and generate much more data. We have prepared for you library with various image altering methods in `BrnoAiDaysWorkshop.Augmentation\AugmentationMethods.cs`. Your task is to use some of these methods on each image to make the datasat **at least three times bigger**.
 
 
## Helpful links
 - [ML.NET documentations](https://learn.microsoft.com/en-us/dotnet/machine-learning/)
 - [Available trainers](https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-choose-an-ml-net-algorithm)
 - [Available transformations](https://learn.microsoft.com/en-us/dotnet/machine-learning/resources/transforms)
 - [Sample codes](https://github.com/dotnet/machinelearning-samples)
