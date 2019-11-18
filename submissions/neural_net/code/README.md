## How to Run
Starting from the directory of the unzipped submission:

```bash
cd code
docker build -t cdc_text_prediction .
docker run -it -v <output_solution_path>:/solution cdc_text_prediction 
```