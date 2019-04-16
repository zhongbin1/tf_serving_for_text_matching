# -*- coding:utf-8 -*-

import json
import requests
import time

textmod = {}
textmod["inputs"] = {"input_left": [[1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     "input_right": [[1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     "keep_prob": 1.0}

textmod["signature_name"] = "predict_cls"
textmod["num"] = 2


header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',"Content-Type": "application/json"}

url='http://localhost:8866/similarity'
#url = 'http://localhost:8501/v1/models/serving_model:predict'

start_time = time.time()
for i in range(1000):
    r = requests.post(url, json=textmod, headers=header_dict, verify=False)

end_time = time.time()
print(r.text)
print("total cost time: {}".format((end_time-start_time)))



