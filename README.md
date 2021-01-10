# Disaster Response Pipeline Project

## Introduction
Follwoing a disaster, agencies receive millions of messages, either directly or through social media. This is when these organizations have limited capacity to analyze and organize this data to filter them and to pull out the most important messages. This categorization is important since each organization is responsible for a different set of issues. 

This project takes the dataset from Figure8 to label to train a model that can automatically classify new incoming messages
in relevant categories. This project can effectively address this task in future disasters. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Open another terminal, run env|grep WORK. You'll see the following output WORKSPACEDOMAIN=udacity-student-workspaces.com WORKSPACEID=view6914b2f4 Now, use the above information to open https://view6914b2f4-3001.udacity-student-workspaces.com/ (general format - https://WORKSPACEID-3001.WORKSPACEDOMAIN/)
