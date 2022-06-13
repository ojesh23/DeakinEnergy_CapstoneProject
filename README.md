# DeakinEnergy SIT782 and SIT764  

## Branching (In case of any doubts, contact Pradeep/Wellia/Seniors)
There are 2 main branches in the repository. 
- master 
- developer

master branch is the one that will be live on Azure/hosting platform. No changes should be pushed/committed to master directly.
developer branch is to be used for making any changes. Any new branch needs to be forked/created from developer branch.

When your changes are complete, then create a Pull Request to merge those changes to master branch. Add code/change-set reviewers as appropriate. 
After the Pull request has been accepted by the reviewers, changes will be merged into master and a new deployment can be triggerred to deploy those changes to Azure/hosting platform.

## Steps to get started 
- Clone the repository using the link from Bitbucket
- Switch to developer branch using Source Tree (or a git client of your choice)
  
## Prefered Steps (Git) 
- Fork the current repository into your repository
- Work on your repository
- when ready, make a pull request and add reviewers to confirm the work (unless it is a small change)

## Data Analysis and Machine Learning Classifiers
- The 'Datasets' folder contains the original data provided to us by tutors/staff. Do not make changes there.
- The 'Data Analysis' folder should be used to commit any files related to deliverables e.g. charts, py notebooks, transformed datasets etc.
- The 'T3' folder under "Data Analysis' contains all the data analysis and ML classifiers work for this trimester (Tri-3, 2020).

## Web App
- The 'Web App' folder is web application made using Flask to integrate ML classifiers.

## Steps to start Web App
- Install required packages and libraries from 'requirements.txt' file.
- Run 'app.py' file to get started.