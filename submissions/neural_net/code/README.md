## How to Run
Starting from the directory of the unzipped submission:

```bash
cd code
docker build -t cdc_text_prediction .
docker run -it -v /home/drew/PycharmProjects/CDC_HardCoded/submissions/neural_net/solution:/solution cdc_text_prediction 
```