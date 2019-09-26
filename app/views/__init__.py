from app import flask_app as app
import json
from datetime import datetime
import numpy as np


@app.route("/heartbeat")
def heartbeat():
    return json.dumps(
        {
            "status": True,
            "service": "Homework_Template",
            "datetime": f"{datetime.now()}"
        }
    )

@app.route("/sum")
def sum(x, y):
    s = x + y
    return json.dump(
        {
            "sum": f"{s}"
        }
    )

@app.route("/minimum")
def minimum(x):
    s = min(x)
    return json.dump(
        {
            "minimum": f"{s}"
        }
    )

@app.route("/product")
def product(x):
    a = np.array(x)
    s = np.prod(a)
    return json.dump(
        {
            "product": f"{s}"
        }
    )




@app.before_first_request
def load_app():
    print("Loading App Before First Request")
