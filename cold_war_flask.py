# -*- coding:utf-8 -*-

import json
from flask import Flask, request
from load_model import Sim_Model

app = Flask(__name__)

model = None

def load_sim_model():
    global model
    model = Sim_Model()


@app.route('/similarity', methods=['POST'])
def calc():
    param = json.loads(request.get_data().decode('utf-8'))
    # input_left = param['query']
    # input_right = param['docs']

    input_left = [[1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    input_right = [[1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    result = model.inference(input_left, input_right)

    data = {'scores': result[0]}

    return str(result[0][0])


if __name__ == '__main__':
    load_sim_model()

    app.run(host='0.0.0.0', port=8866, debug=True)

