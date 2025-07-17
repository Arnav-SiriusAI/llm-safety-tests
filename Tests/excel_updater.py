
import pandas as pd
import os
from datetime import datetime
from threading import Lock

file_lock = Lock()

def log_results(llm_name: str, results_dict: dict, output_file: str = 'LLM_Safety_Tests.xlsx'):
    """
    Logs one or more results for a specific LLM to an Excel sheet.

    It automatically updates the 'Date and Time' for the corresponding row.

    Args:
        llm_name (str): The name of the model in the 'LLM Tested' column.
        results_dict (dict): A dictionary where keys are the column names and values 
                             are the scores to log.
        output_file (str): The path to your Excel file.
    """
    with file_lock:
        primary_key = 'LLM Tested'
        
        if os.path.exists(output_file):
            df = pd.read_excel(output_file, index_col=primary_key)
        else:
            df = pd.DataFrame().set_index(pd.Index([], name=primary_key))

        # Create a new dictionary for all updates, starting with the timestamp
        updates_to_log = {'Date and Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        # Add all the results from the passed-in dictionary
        updates_to_log.update(results_dict)

        # Update each value for the specified LLM
        for column, value in updates_to_log.items():
            if column not in df.columns:
                df[column] = pd.NA
            
            # .loc will find the row and column and set the value
            df.loc[llm_name, column] = value

        df.reset_index(inplace=True)
        df.to_excel(output_file, index=False)
        
    print(f"✅ Logged {len(results_dict)} result(s) for '{llm_name}' to Excel.")
