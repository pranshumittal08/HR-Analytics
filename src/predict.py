import joblib
import os
import pandas as pd
import config

if __name__ == "__main__":
    test_df = pd.read_csv(config.test_df, index_col = "employee_id")
    
    model = joblib.load(os.path.join(config.model_path,"stack_model_catb_1.pkl"))
    
    test_pred = model.predict(test_df)
    
    test_results = pd.DataFrame(test_pred, index = test_df.index, columns = ["is_promoted"])
    
    test_results.to_csv("C:\\Users\\prans\\Python files\\HR Analytics\\submissions\\test_result.csv")
                        
                        