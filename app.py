import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle
from flask import jsonify

app = Flask(__name__)

filename = 'lr_model.pkl'
with open(filename, 'rb') as file:  
    model = pickle.load(file)

@app.route('/sales_data',methods=['POST'])
def sales_data():
    req_data = request.get_json()
    print(req_data)
    print(type(req_data))
    
    df = pd.DataFrame(columns = ['date_block_num','item_cnt_day'])
    df = df.append(req_data)
    print(df)

    df = df.groupby(['date_block_num']).agg({'item_cnt_day':'sum'})
    df.reset_index(inplace=True)
    df.rename({'item_cnt_day':'item_cnt_month'},axis=1,inplace=True)
    print(df)

    date_block_num_list = df['date_block_num'].unique()
    date_block_num_list.sort()
    print(list(date_block_num_list))

    for i in range(1,34):
        if i not in list(date_block_num_list):
            df = df.append({'date_block_num' : i},ignore_index=True)

    print(df)
    df = df[df['date_block_num']!=0]
    print(df)


    df.replace(np.NaN,0,inplace=True)
    df['item_cnt_month']=df['item_cnt_month'].clip(0,30)

    df.sort_values(['date_block_num'],inplace=True)
    df.reset_index(inplace=True,drop=True)
    print(df)

    sales = df['item_cnt_month'].to_numpy()
    sales = sales.reshape(1,33)
    print(sales)

    predictions = model.predict(sales)
    predictions = predictions.round()
    predictions = predictions.clip(0,30)
    predictions = predictions[0]
    print(predictions)
    print(type(predictions))
    print(predictions.shape)
    print("DONE")

    return jsonify(
        date_block_num = 34,
        item_cnt_month = predictions
    )
                      



if __name__ == '__main__':
    app.run(debug=True)