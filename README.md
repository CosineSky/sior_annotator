
## Tips for running the pipeline
**0. Preparing dataset**  
Put SIOR dataset `data/`, make sure at least dirs below exist:
     ```
     data/trainval_images/
     data/test_images/
     data/semlabels/gray
     data/train.txt
     data/test.txt
     ```
   

**1. (Optional) Setting LLM API keys**
If LLM-assisted methods are used in mask cleaning, set the API keys in `configs/api_key.py`.
   

**2. Running the pipeline**  
   ```bash
   python stages-extra/run_pipeline_extra.py
   ```
   
**3. Results**  
   The  results will be saved in `logs/` and `output/` directories. MIOUS value after a rule-only mask cleaning process is expected to be around 0.7.

   
   
