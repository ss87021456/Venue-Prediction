# Venue-Prediction
#### Venue Prediction with bag-of-words + heterogenous information as features using sklearn SGDClassifier

## Dataset DBLP:
training: https://www.dropbox.com/s/rrbksqvvoefrr4p/training.txt?dl=0
<br>validation: https://www.dropbox.com/s/tw094y2xfcoosv3/validation.txt?dl=0

## Dependency:
python3
<br> sklearn
<br> pandas
<br> numpy
<br> pickle

## Pipeline:
mkdir input # Create input directory
<br> <Download training, validation dataset on the link above and move into input directory>
<br>python3 ./src/clean_data.py --input ./input/training.txt --output ./input/cleaned_training.txt
<br>python3 ./src/clean_data.py --input ./input/validation.txt --output ./input/cleaned_training.txt
<br>python3 ./src/create_data_example.py --train ./input/cleaned_training.txt --validation ./input/cleaned_validation.txt
<br>python3 ./src/train_classifier.py --train ./input/cleaned_training.txt --validation ./input/cleaned_validation.txt

## Default Configuration:
bag-of-word dimension: 3000
<br>classifier: sklearn SGDClassifier (default)

## Result: (on validation dataset)

| Feature | F1-micro | F1-macro | Accuracy |
| --- | --- | --- | --- |
| title info. | 0.266 | 0.172 | 0.267 |
| title + cited_venue info. | 0.982 | 0.758 | 0.981 |

