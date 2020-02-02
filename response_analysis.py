from flask import request
from flask_restful import Resource

from analyzer import analyze


class ResponseAnalysis(Resource):
    def post(self):
        text = request.json['text']
        return {
            'analysis': analyze(text),
            'text': text
        }
