from flask import Flask
from flask_restful import Resource, Api

from response_analysis import ResponseAnalysis

app = Flask(__name__)
api = Api(app)


api.add_resource(ResponseAnalysis, '/analysis')

if __name__ == '__main__':
    app.run(debug=True)
