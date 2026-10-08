from gurulearn import QAAgent
import pandas as pd
data=pd.read_csv("data.csv")
model=QAAgent(
    data=data,
    llm_model="llama3.2",
    page_content_fields=["review_id","restaurant_name","state_level_restaurant","district","state","country","rating","review_text"],
    metadata_fields=["review_text","popular_dish","expensive_dish","cheap_dish","most_ordered_dish"],
    system_prompt="you are a helpful sentiment analysis assistant that classifies reviews and give better output for the customers"
    
)
print(model.query("what is the best shop there"))
