This is a regression model to predict stident performance.

Dataset: https://www.kaggle.com/datasets/devansodariya/student-performance-data/data

Original data from UCI: https://archive.ics.uci.edu/dataset/320/student+performance

The description of the dataset from the Kaggle: Student Performance Data was obtained in a survey of students' math course in secondary school. It consists of 33 columns. Dataset Contains Features like:
- school ID
- gender
- age
- size of family
- Father education
- Mother education
- Occupation of Father and Mother
- Family Relation
- Health
- Grades

To run the project, use the Pipfile and Pipfile.lock to load the dependencies. I have used python 3.11 and scikit-learn==1.5.2, flask, waitress, also xgboost for the final prediction. In cmd, after initializing the virtual environment, run the predict.py:

python predict.py

In the new terminal run the predict_test.py:

python predict_test.py

To deploy in Docker, start it and run in cmd:

docker build -t student-performance .

docker run -it -p 9696:9696 student-performance:latest

Then open another cmd and run: 

python predict_test.py

The images of the last two steps are loaded into the folder (Image_1 and Image_2).

